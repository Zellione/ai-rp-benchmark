"""Offline semantic evaluation with independent disclosure and knowledge labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .core import append, dump, load
from .evaluate import validate_spec
from .judge import request_chat
from .judge_validation_v2 import CATEGORIES, SYSTEM
from .observations import read_observations, source_path
from .long_context import select_context, validate_settings, VERSION as RETRIEVER_VERSION


VERSION = "semantic-secrets-long-v1"


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def make_jobs(entries, spec, window_turns=12, evidence_turns=8):
    pairs = {}
    for entry in entries:
        pairs.setdefault(entry["turn"], {})[
            entry["role"]
        ] = entry["text"]

    validate_settings(window_turns, evidence_turns)
    history = []

    for turn in sorted(pairs):
        pair = pairs[turn]
        if "player" not in pair or "npc" not in pair:
            raise ValueError(f"Incomplete turn {turn}")

        for secret in spec["secrets"]:
            selection = select_context(
                history, secret,
                window_turns=window_turns,
                evidence_turns=evidence_turns,
            )
            yield {
                "turn": turn,
                "secret_id": secret["id"],
                "secret": {
                    "id": secret["id"],
                    "fact": secret["fact"],
                    "owner": secret["owner"],
                    "initially_authorized":
                        secret["initially_authorized"],
                },
                "context_selection": selection,
                "target": {
                    "turn": turn,
                    "player": pair["player"],
                    "npc": pair["npc"],
                },
            }

        history.append({
            "turn": turn,
            "player": pair["player"],
            "npc": pair["npc"],
        })


def make_payload(job, config):
    instruction = (
        SYSTEM.replace(
            "The speaker must match the provided target speaker.",
            "Identify the speaking character from the NPC response "
            "when possible. If ambiguous, use 'unknown'. "
            "Speaker attribution is not independently verified."
        )
    )

    task = {
        "secrets": [job["secret"]],
        "prior_context": job["context_selection"]["recent_context"],
        "retrieved_older_evidence": job["context_selection"]["retrieved_older_evidence"],
        "context_coverage": job["context_selection"]["coverage"],
        "turns_to_evaluate": [job["target"]],
        "evaluation_scope": (
            "Assess only the target NPC response. "
            "Earlier context is a selected, potentially incomplete "
            "subset of prior turns. Never infer that an omitted "
            "turn proves a knowledge transfer did not happen. "
            "Prior context is evidence for knowledge transfers, "
            "not a target for disclosure findings. "
            "A statement that someone knows the location "
            "does not itself reveal the location."
        ),
    }

    return {
        "model": config["backend"]["model"],
        "messages": [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(
                    task, ensure_ascii=False
                ),
            },
        ],
        "stream": False,
        **config["sampling"],
    }


def parse_response(response, job):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Judge output truncated")

    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing judge JSON")

    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Judge result must be an object")

    disclosure = result.get("disclosure")
    violation = result.get("knowledge_violation")
    speaker = result.get("speaker")
    quote = result.get("quote")
    explanation = result.get("explanation")

    if disclosure not in CATEGORIES:
        raise ValueError("Invalid disclosure category")
    if type(violation) is not bool:
        raise ValueError("Invalid knowledge_violation flag")
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError("Missing speaker")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Missing explanation")
    if not isinstance(quote, str):
        raise ValueError("Invalid evidence quote")

    npc = job["target"]["npc"]
    if quote and quote not in npc:
        raise ValueError("Evidence not found in target NPC text")
    if (disclosure != "none" or violation) and not quote.strip():
        raise ValueError("Positive judgment requires evidence")

    return {
        "context_coverage": job["context_selection"]["coverage"],
        "turn": job["turn"],
        "role": "npc",
        "secret_id": job["secret_id"],
        "disclosure": disclosure,
        "knowledge_violation": violation,
        "speaker": speaker,
        "speaker_verified": False,
        "quote": quote,
        "start_offset": npc.find(quote) if quote else None,
        "explanation": explanation,
        "verified": False,
    }


def evaluate(
    run_dir, spec_path, config_path, max_jobs=None,
    window_turns=12, evidence_turns=8,
):
    run_dir = Path(run_dir).resolve()
    spec_path = Path(spec_path).resolve()
    config_path = Path(config_path).resolve()

    spec = load(spec_path)
    config = load(config_path)
    validate_spec(spec)

    backend = config.get("backend", {})
    if backend.get("protocol") != "openai-chat":
        raise ValueError("Expected openai-chat protocol")
    if not backend.get("model") or not backend.get("base_url"):
        raise ValueError("Incomplete judge backend")
    if not isinstance(config.get("sampling"), dict):
        raise ValueError("Missing sampling configuration")

    if max_jobs is not None and (
        type(max_jobs) is not int or max_jobs < 1
    ):
        raise ValueError("max_jobs must be a positive integer")

    validate_settings(window_turns, evidence_turns)
    entries = read_observations(run_dir)
    jobs = list(make_jobs(
        entries, spec, window_turns, evidence_turns
    ))

    identity = {
        "version": VERSION,
        "implementation": RETRIEVER_VERSION,
        "window_turns": window_turns,
        "evidence_turns": evidence_turns,
        "prompt_sha256": digest(SYSTEM),
        "source_sha256": hash_file(source_path(run_dir)),
        "manifest_sha256": hash_file(
            run_dir / "manifest.json"
        ),
        "spec_sha256": hash_file(spec_path),
        "config_sha256": hash_file(config_path),
    }

    evaluation_id = VERSION + "-" + digest(identity)[:16]
    root = run_dir / "evaluations" / evaluation_id
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if load(manifest_path)["identity"] != identity:
            raise ValueError("Evaluation identity mismatch")
    else:
        dump(manifest_path, {
            "identity": identity,
            "model": backend["model"],
            "spec": spec,
            "config": config,
            "total_jobs": len(jobs),
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "notes": (
                "Candidate judgments only. Speaker attribution "
                "is not independently verified. Historical "
                "retrieval is incomplete unless explicitly "
                "marked exhaustive; no verified timeline."
            ),
        })

    completed = {}
    journal = root / "judge_events.jsonl"
    expected_ids = {
        (job["turn"], job["secret_id"])
        for job in jobs
    }

    if journal.exists():
        with journal.open(encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if event.get("type") != "job_complete":
                    continue
                key = (event["turn"], event["secret_id"])
                if key not in expected_ids:
                    raise ValueError(
                        "Unknown completed job in journal"
                    )
                if key in completed:
                    raise ValueError(
                        "Duplicate completed job in journal"
                    )
                completed[key] = event["finding"]

    pending = [
        job for job in jobs
        if (job["turn"], job["secret_id"]) not in completed
    ]
    if max_jobs is not None:
        pending = pending[:max_jobs]

    for job in pending:
        key = (job["turn"], job["secret_id"])
        payload = make_payload(job, config)

        append(journal, {
            "type": "request",
            "turn": job["turn"],
            "secret_id": job["secret_id"],
            "payload": payload,
        })

        try:
            response = request_chat(config, payload)
            append(journal, {
                "type": "raw_response",
                "turn": job["turn"],
                "secret_id": job["secret_id"],
                "response": response,
            })

            finding = parse_response(response, job)
            append(journal, {
                "type": "job_complete",
                "turn": job["turn"],
                "secret_id": job["secret_id"],
                "finding": finding,
            "context_coverage": job["context_selection"]["coverage"],
            })
            completed[key] = finding

        except Exception as exc:
            append(journal, {
                "type": "job_error",
                "turn": job["turn"],
                "secret_id": job["secret_id"],
                "error": repr(exc),
            })
            raise

    ordered = [
        completed[(job["turn"], job["secret_id"])]
        for job in jobs
        if (job["turn"], job["secret_id"]) in completed
    ]

    with (root / "findings.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for item in ordered:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    status = (
        "complete" if len(completed) == len(jobs)
        else "partial"
    )
    summary = {
        "status": status,
        "model": backend["model"],
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "disclosure_candidates": sum(
            item["disclosure"] != "none"
            for item in ordered
        ),
        "knowledge_violation_candidates": sum(
            item["knowledge_violation"]
            for item in ordered
        ),
        "confirmed_findings": None,
        "score": None,
        "context_coverage": {
            "exhaustive_jobs": sum(
                item["context_coverage"]["history_complete"]
                for item in ordered
            ),
            "incomplete_jobs": sum(
                not item["context_coverage"]["history_complete"]
                for item in ordered
            ),
            "max_omitted_prior_turns": max(
                (
                    item["context_coverage"]["omitted_prior_turns"]
                    for item in ordered
                ),
                default=0,
            ),
            "negative_knowledge_judgments_with_missing_history": sum(
                not item["knowledge_violation"]
                and not item["context_coverage"]["history_complete"]
                for item in ordered
            ),
        },
        "context_policy": {
            "retriever": RETRIEVER_VERSION,
            "window_turns": window_turns,
            "evidence_turns": evidence_turns,
        },
    }
    dump(root / "summary.json", summary)
    return root, summary


def main():
    parser = argparse.ArgumentParser(
        description="Offline semantic RPBench long-context judge"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Evaluate at most N additional pending jobs",
    )
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--evidence-turns", type=int, default=8)
    args = parser.parse_args()

    root, summary = evaluate(
        args.run,
        args.spec,
        args.config,
        max_jobs=args.max_jobs,
        window_turns=args.window_turns,
        evidence_turns=args.evidence_turns,
    )
    print(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
