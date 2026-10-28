"""Dimension-wise, position-balanced RP quality comparisons."""

import argparse
import hashlib
import itertools
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Backend
from .evaluator_llm import parse_evaluator_json
from .rp_pairwise_judge import validate_dataset
from .rp_quality_judge import DIMENSIONS


CRITERIA = {
    "german_prose": (
        "Compare natural, idiomatic German prose, grammar, "
        "word choice, rhythm, imagery, and stylistic variety. "
        "Do not reward verbosity or decorative language alone."
    ),
    "character_fidelity": (
        "Compare fidelity to the NPC's personality, motivations, "
        "knowledge boundaries, voice, and credible character "
        "development. Respect plausible secrecy."
    ),
    "dialogue_responsiveness": (
        "Compare how meaningfully the NPC responds to the "
        "player's actual dialogue, questions, objections, "
        "actions and changing behavior."
    ),
    "subtext": (
        "Compare emotional credibility, restraint, tension, "
        "implicit meaning and psychological nuance. Penalize "
        "unnecessary explanation of emotions."
    ),
    "narrative_initiative": (
        "Compare meaningful NPC contributions and natural "
        "development of the scene, without railroading or "
        "controlling the player."
    ),
    "continuity": (
        "Compare spatial, temporal, causal, object, relationship "
        "and character-memory consistency across the dialogue. "
        "Notice concrete contradictions."
    ),
}

assert set(CRITERIA) == set(DIMENSIONS)

SYSTEM = """You are an independent judge of German literary RP.

You receive two complete conversations and ONE evaluation criterion.
Evaluate ONLY the NPC. The player text provides context and is not
itself being scored.

Focus exclusively on the requested criterion. Evaluate each NPC
in the context of its own unfolding conversation. Do not reward a
different plot, faster revelations, or a more cooperative player.

Choose X, Y, or tie. Choose tie whenever the available evidence
does not justify a meaningful preference. Both NPCs may have
strengths and weaknesses.

Return only JSON:
{
  "winner": "tie",
  "reason": "Concise comparative reasoning",
  "evidence_x": [
    {"turn": 1, "quote": "Exact NPC excerpt"}
  ],
  "evidence_y": [
    {"turn": 1, "quote": "Exact NPC excerpt"}
  ]
}

Cite 1 or 2 short, exact NPC excerpts per conversation.
Use their actual NPC turn numbers. Never paraphrase a quotation.
Do not infer the model identity, backend, or quantization.
Do not include other fields.
"""

FIELDS = {"winner", "reason", "evidence_x", "evidence_y"}


def jobs_for(dataset, dimensions, seed, selected_pair=None):
    ids = sorted(
        case["id"] for case in validate_dataset(dataset)["cases"]
    )
    if (
        selected_pair is not None
        and (
            len(selected_pair) != 2
            or selected_pair[0] == selected_pair[1]
            or any(item not in ids for item in selected_pair)
        )
    ):
        raise ValueError("Invalid candidate pair")

    if not dimensions or any(
        dimension not in CRITERIA for dimension in dimensions
    ) or len(set(dimensions)) != len(dimensions):
        raise ValueError("Invalid dimensions")

    if selected_pair is None:
        pairs = list(itertools.combinations(ids, 2))
    else:
        pairs = [tuple(sorted(selected_pair))]

    jobs = [
        (x, y, criterion)
        for a, b in pairs
        for x, y in ((a, b), (b, a))
        for criterion in dimensions
    ]
    random.Random(seed).shuffle(jobs)
    return jobs


def validate_decision(value, x, y):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Invalid decision fields")
    if value["winner"] not in {"X", "Y", "tie"}:
        raise ValueError("Invalid winner")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Missing reason")

    for field, case in (
        ("evidence_x", x),
        ("evidence_y", y),
    ):
        evidence = value[field]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 2:
            raise ValueError(f"Invalid {field}")

        npc_turns = {
            item["turn"]: item["text"]
            for item in case["transcript"]
            if item["role"] == "npc"
        }
        for item in evidence:
            if (
                not isinstance(item, dict)
                or set(item) != {"turn", "quote"}
                or type(item["turn"]) is not int
                or not isinstance(item["quote"], str)
                or not item["quote"].strip()
                or item["quote"] not in npc_turns.get(item["turn"], "")
            ):
                raise ValueError(f"Unsupported NPC evidence: {field}")

    return value


def make_backend(config):
    backend = config["backend"]
    if backend.get("mode", "chat") != "chat":
        raise ValueError("Judge currently requires chat mode")
    if backend["name"] in {"ninfer", "strata"} and not backend.get(
        "verified_openai_api", False
    ):
        raise ValueError("Unverified judge backend API")

    return Backend(
        name=backend["name"],
        base_url=backend["base_url"],
        model=backend["model"],
        mode="chat",
        api_key=os.getenv(
            backend.get("api_key_env", "RPBENCH_JUDGE_API_KEY")
        ),
        timeout=int(backend.get("timeout", 900)),
    )


def evaluate(x, y, criterion, context, backend, sampling, retry_budget):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({
            "criterion": criterion,
            "criterion_definition": CRITERIA[criterion],
            "npc_instructions": context,
            "conversation_X": x["transcript"],
            "conversation_Y": y["transcript"],
        }, ensure_ascii=False)},
    ]

    record = {"valid": False, "attempts": []}
    budgets = [sampling["max_tokens"]]
    if retry_budget is not None:
        budgets.append(retry_budget)

    for index, budget in enumerate(budgets):
        request_sampling = {**sampling, "max_tokens": budget}
        attempt = {
            "max_tokens": budget,
            "content": None,
            "reasoning": None,
            "finish_reason": None,
            "usage": None,
        }
        start = time.perf_counter()

        try:
            url, payload = backend.prepare(
                messages, request_sampling, {}
            )
            content, response = backend.generate(url, payload)
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}

            attempt.update({
                "content": content,
                "reasoning": (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
            })

            if attempt["finish_reason"] == "length":
                raise ValueError("Generation truncated")

            verdict = validate_decision(
                parse_evaluator_json(content), x, y
            )
            record.update({"valid": True, "verdict": verdict})
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            attempt["elapsed_seconds"] = round(
                time.perf_counter() - start, 3
            )
            record["attempts"].append(attempt)

        if record["valid"]:
            break

        # Increase the budget ONLY after a token-limit truncation.
        if (
            index == 0
            and retry_budget is not None
            and attempt["finish_reason"] == "length"
        ):
            continue
        break

    return record



def job_key(x, y, criterion):
    return (x, y, criterion)


def plan_jobs(jobs, previous, fingerprints, seed, retry_failed=False):
    """Resume without repeating accepted jobs or silent re-evaluation."""
    successful = set()
    attempted = set()

    for record in previous:
        if not isinstance(record, dict):
            raise ValueError("Invalid archived job")

        for field, digest in fingerprints.items():
            if record.get(field) != digest:
                raise ValueError(
                    f"Archived {field} differs from current inputs"
                )

        if record.get("seed") != seed:
            raise ValueError("Archived seed differs")

        key = job_key(
            record.get("x"),
            record.get("y"),
            record.get("criterion"),
        )

        if key not in jobs:
            raise ValueError("Archived job not in current plan")

        attempted.add(key)
        if record.get("valid") is True:
            successful.add(key)

    pending = [
        key for key in jobs
        if key not in successful
        and (retry_failed or key not in attempted)
    ]
    return pending


def read_archive(filename):
    if not filename.exists():
        return []

    records = []
    for number, line in enumerate(
        filename.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid archived JSONL line {number}"
            ) from error

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"))
    parser.add_argument("--dimensions", nargs="+", default=list(CRITERIA))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be 1..3600 seconds")

    data_bytes = Path(args.dataset).read_bytes()
    data = validate_dataset(json.loads(data_bytes))
    config_bytes = Path(args.judge_config).read_bytes()
    config = json.loads(config_bytes)
    scenario_bytes = Path(args.scenario).read_bytes()
    context = json.loads(scenario_bytes)["system"]

    backend = make_backend(config)
    backend.timeout = args.timeout

    sampling = config["sampling"]
    base_budget = sampling["max_tokens"]
    retry_budget = config.get("retry_max_tokens")

    if (
        type(base_budget) is not int or base_budget <= 0
        or (
            retry_budget is not None
            and (
                type(retry_budget) is not int
                or retry_budget <= base_budget
            )
        )
    ):
        parser.error("Invalid initial/retry token budget")

    try:
        jobs = jobs_for(
            data, args.dimensions, args.seed, args.pair
        )
    except ValueError as exc:
        parser.error(str(exc))

    fingerprints = {
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
    }

    cases = {case["id"]: case for case in data["cases"]}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    previous = read_archive(output)

    try:
        pending = plan_jobs(
            jobs, previous, fingerprints, args.seed,
            retry_failed=args.retry_failed,
        )
    except ValueError as error:
        parser.error(str(error))

    print(
        f"Planned={len(jobs)} archived={len(previous)} "
        f"pending={len(pending)} skipped={len(jobs)-len(pending)}",
        flush=True,
    )

    if not pending:
        print("No pending jobs.")
        return

    # Append-only, including failed jobs. Keep previous attempts.
    with output.open("a", encoding="utf-8") as file:
        for i, (xid, yid, criterion) in enumerate(pending, 1):
            print(
                f"{i:02}/{len(pending):02} | "
                f"{xid} vs {yid} | {criterion}",
                flush=True,
            )

            result = evaluate(
                cases[xid], cases[yid], criterion,
                context, backend, sampling, retry_budget,
            )
            result.update({
                "x": xid,
                "y": yid,
                "criterion": criterion,
                "seed": args.seed,
                "judge_model": backend.model,
                "request_timeout_seconds": args.timeout,
                **fingerprints,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

            file.write(
                json.dumps(result, ensure_ascii=False) + "\\n"
            )
            file.flush()
            os.fsync(file.fileno())

            winner = (
                result["verdict"]["winner"]
                if result["valid"] else "INVALID"
            )
            print(
                f"  {winner} | valid={result['valid']} "
                f"| attempts={len(result['attempts'])}",
                flush=True,
            )

    print("Saved:", output)


if __name__ == "__main__":
    main()
