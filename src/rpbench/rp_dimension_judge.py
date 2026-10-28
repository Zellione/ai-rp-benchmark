"""Independent RP quality assessment: one conversation, one dimension."""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Backend
from .evaluator_llm import parse_evaluator_json
from .rp_pairwise_judge import validate_dataset
from .rp_pairwise_dimensions import CRITERIA


INSTRUCTIONS = """You are an independent evaluator of German literary roleplay.

Evaluate ONE complete conversation for ONE specified quality dimension.
Evaluate only the NPC's writing and behavior. Player messages provide
context but are not themselves being scored.

Use the supplied NPC character and scenario instructions.
Consider all NPC turns, including strengths, weaknesses and mistakes.
Do not reward verbosity, melodrama or premature plot revelations.
Do not penalize the NPC merely for keeping a character secret.

Use the complete 1-10 scale. Scores must reflect the specified
dimension, not an overall impression of the conversation.

Return ONLY this JSON object:

{
  "score": 7,
  "reason": "Concise evidence-based explanation",
  "evidence": [
    {"turn": 1, "quote": "Exact excerpt from an NPC turn"}
  ]
}

Evidence requirements:
- Provide one or two brief, exact NPC quotations.
- Cite the actual NPC turn numbers.
- Never quote player text as NPC evidence.
- Never paraphrase a quotation.
- No additional fields.
- Make a decision and return the JSON once evidence is sufficient.
"""


def validate_verdict(value, transcript):
    if not isinstance(value, dict) or set(value) != {
        "score", "reason", "evidence"
    }:
        raise ValueError("Invalid verdict fields")

    if type(value["score"]) is not int or not 1 <= value["score"] <= 10:
        raise ValueError("Score must be an integer from 1 to 10")

    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Missing reason")

    evidence = value["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 2:
        raise ValueError("Expected one or two evidence excerpts")

    npc_turns = {
        item["turn"]: item["text"]
        for item in transcript
        if item["role"] == "npc"
    }

    for citation in evidence:
        if (
            not isinstance(citation, dict)
            or set(citation) != {"turn", "quote"}
            or type(citation["turn"]) is not int
            or not isinstance(citation["quote"], str)
            or not citation["quote"].strip()
            or citation["quote"] not in npc_turns.get(
                citation["turn"], ""
            )
        ):
            raise ValueError("Unsupported NPC evidence")

    return value


def create_backend(config):
    cfg = config["backend"]

    if cfg.get("mode", "chat") != "chat":
        raise ValueError("Judge requires chat mode")

    if cfg["name"] in {"ninfer", "strata"} and not cfg.get(
        "verified_openai_api", False
    ):
        raise ValueError("Unverified judge API")

    return Backend(
        name=cfg["name"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        mode="chat",
        api_key=os.getenv(
            cfg.get("api_key_env", "RPBENCH_JUDGE_API_KEY")
        ),
        timeout=int(cfg.get("timeout", 300)),
    )


def evaluate(case, dimension, context, backend, sampling):
    if dimension not in CRITERIA:
        raise ValueError("Unknown dimension")

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": json.dumps({
            "dimension": dimension,
            "dimension_definition": CRITERIA[dimension],
            "npc_instructions": context,
            "conversation": case["transcript"],
        }, ensure_ascii=False)},
    ]

    record = {
        "valid": False,
        "content": None,
        "reasoning": None,
        "usage": None,
        "finish_reason": None,
    }

    started = time.perf_counter()

    try:
        url, payload = backend.prepare(messages, sampling, {})
        content, response = backend.generate(url, payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        record.update({
            "content": content,
            "reasoning": (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            ),
            "usage": response.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        })

        if record["finish_reason"] == "length":
            raise ValueError("Judge generation truncated")

        verdict = validate_verdict(
            parse_evaluator_json(content),
            case["transcript"],
        )
        record["verdict"] = verdict
        record["valid"] = True

    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"

    record["elapsed_seconds"] = round(
        time.perf_counter() - started, 3
    )
    return record


def plan_jobs(dataset, dimensions, candidates, seed):
    valid = validate_dataset(dataset)
    ids = {case["id"] for case in valid["cases"]}

    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate or missing dimensions")
    if any(item not in CRITERIA for item in dimensions):
        raise ValueError("Unknown dimension")

    if candidates is None:
        candidates = sorted(ids)

    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(item not in ids for item in candidates)
    ):
        raise ValueError("Invalid candidate selection")

    jobs = [
        (cid, dimension)
        for cid in sorted(candidates)
        for dimension in dimensions
    ]
    random.Random(seed).shuffle(jobs)
    return jobs


def pending_jobs(jobs, records, fingerprints, retry_failed=False):
    planned = set(jobs)
    attempted = set()
    successful = set()

    for record in records:
        for field, expected in fingerprints.items():
            if record.get(field) != expected:
                raise ValueError(f"Archive mismatch: {field}")

        key = (record.get("candidate"), record.get("dimension"))
        if key not in planned:
            raise ValueError("Archived job not in current plan")

        attempted.add(key)
        if record.get("valid") is True:
            successful.add(key)

    return [
        job for job in jobs
        if job not in successful
        and (retry_failed or job not in attempted)
    ]


def read_archive(filename):
    if not filename.exists():
        return []

    records = []
    for line_number, line in enumerate(
        filename.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}"
                ) from error
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--dimensions", nargs="+", default=list(CRITERIA))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    dataset_bytes = Path(args.dataset).read_bytes()
    dataset = json.loads(dataset_bytes)
    scenario_bytes = Path(args.scenario).read_bytes()
    context = json.loads(scenario_bytes)["system"]
    config_bytes = Path(args.judge_config).read_bytes()
    config = json.loads(config_bytes)

    backend = create_backend(config)
    sampling = config["sampling"]

    from .sampling import validate_sampling
    validate_sampling(backend.name, backend.mode, sampling)

    try:
        jobs = plan_jobs(
            dataset, args.dimensions, args.candidates, args.seed
        )
    except ValueError as error:
        parser.error(str(error))

    fingerprints = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        archived = read_archive(output)
        pending = pending_jobs(
            jobs, archived, fingerprints,
            retry_failed=args.retry_failed,
        )
    except ValueError as error:
        parser.error(str(error))

    print(
        f"Planned={len(jobs)} archived={len(archived)} "
        f"pending={len(pending)}",
        flush=True,
    )

    cases = {
        case["id"]: case
        for case in dataset["cases"]
    }

    with output.open("a", encoding="utf-8") as file:
        for index, (cid, dimension) in enumerate(pending, 1):
            print(
                f"{index:02}/{len(pending):02} | "
                f"{cid} | {dimension}",
                flush=True,
            )

            record = evaluate(
                cases[cid], dimension,
                context, backend, sampling,
            )
            record.update({
                "candidate": cid,
                "dimension": dimension,
                "judge_model": backend.model,
                "sampling": sampling,
                **fingerprints,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            file.flush()
            os.fsync(file.fileno())

            print(
                f"  valid={record['valid']} "
                f"score={(record.get('verdict') or {}).get('score')} "
                f"tokens={(record.get('usage') or {}).get('completion_tokens')} "
                f"time={record['elapsed_seconds']}s",
                flush=True,
            )

    print("Saved:", output)


if __name__ == "__main__":
    main()
