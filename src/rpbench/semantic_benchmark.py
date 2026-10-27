"""Standalone semantic adjudicator benchmark (no RP journal writes)."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import load
from .criteria_runtime import _generate
from .semantic_criteria import validate_semantic_proposal
from .state import initial_state


def validate_suite(suite):
    if (
        not isinstance(suite, dict)
        or suite.get("schema_version") != 1
        or not isinstance(suite.get("cases"), list)
        or not suite["cases"]
    ):
        raise ValueError("Invalid benchmark suite")

    ids = set()

    for case in suite["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Invalid benchmark case")
        for field in (
            "id", "goal_id", "criterion_id",
            "description", "expected_status",
        ):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"Missing {field}")

        if case["id"] in ids:
            raise ValueError("Duplicate benchmark case")
        ids.add(case["id"])

        if case["expected_status"] not in {
            "satisfied", "partial", "not_satisfied"
        }:
            raise ValueError("Invalid expected status")

        observations = case.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("Missing observations")

        turns = {}
        for item in observations:
            if not isinstance(item, dict) or set(item) != {
                "turn", "role", "text"
            }:
                raise ValueError("Invalid observation")
            turn = item["turn"]
            role = item["role"]
            if (
                type(turn) is not int or turn < 1
                or role not in {"player", "npc"}
                or not isinstance(item["text"], str)
            ):
                raise ValueError("Invalid observation value")
            if role in turns.setdefault(turn, {}):
                raise ValueError("Duplicate observation role")
            turns[turn][role] = item["text"]

        if (
            sorted(turns) != list(range(1, max(turns) + 1))
            or any(set(entries) != {"player", "npc"}
                   for entries in turns.values())
        ):
            raise ValueError("Missing turn observation")

        partials = case.get("prior_partials", [])
        if not isinstance(partials, list):
            raise ValueError("Invalid prior partials")
        for partial in partials:
            if (
                not isinstance(partial, dict)
                or set(partial) != {"turn", "reason"}
                or type(partial["turn"]) is not int
                or partial["turn"] not in turns
                or partial["turn"] >= max(turns)
                or not isinstance(partial["reason"], str)
                or not partial["reason"]
            ):
                raise ValueError("Invalid prior partial")

        sources = case.get("required_source_turns", [])
        if (
            not isinstance(sources, list)
            or any(type(t) is not int or t not in turns
                   for t in sources)
        ):
            raise ValueError("Invalid required sources")

    return suite


def case_inputs(case):
    gid = case["goal_id"]
    cid = case["criterion_id"]

    truth = {
        "schema_version": 1,
        "player": {"subgoals": {
            gid: {
                "description": case["description"],
                "initial_status": "pending",
            }
        }},
        "characters": {},
    }

    definition = {
        "schema_version": 1,
        "goals": {gid: {
            "criteria": [{
                "id": cid,
                "description": case["description"],
                "evidence_contains": "benchmark-not-used",
            }],
        }},
    }

    progress = {"goals": {gid: {"criteria": {cid: {
        "description": case["description"],
        "proposed_satisfied": False,
        "evidence": [
            {
                "turn": part["turn"],
                "status": "partial",
                "reason": part["reason"],
            }
            for part in case.get("prior_partials", [])
        ],
    }}}}}

    return definition, initial_state(truth), progress


def run_case(case, settings):
    definition, state, progress = case_inputs(case)
    captured = {}
    start = time.perf_counter()
    record = {
        "case": case["id"],
        "expected_status": case["expected_status"],
        "required_source_turns": case.get(
            "required_source_turns", []
        ),
    }

    try:
        proposal, metadata = _generate(
            definition, progress,
            case["observations"], settings,
            goal_id=case["goal_id"],
            criterion_ids=[case["criterion_id"]],
            on_response=captured.update,
        )

        current_turn = max(
            item["turn"] for item in case["observations"]
        )
        result = validate_semantic_proposal(
            proposal, definition, state,
            case["observations"], current_turn,
        )
        updates = result["criterion_updates"]

        if (
            len(updates) != 1
            or updates[0]["goal_id"] != case["goal_id"]
            or updates[0]["criterion_id"] != case["criterion_id"]
        ):
            raise ValueError("Wrong number or target of updates")

        update = updates[0]
        if metadata.get("finish_reason") == "length":
            raise ValueError("Truncated generation")

        sources = update.get("source_turns", [])
        required = set(record["required_source_turns"])

        record.update({
            "valid": True,
            "status": update["status"],
            "source_turns": sources,
            "reason": update.get("reason"),
            "correct": (
                update["status"] == case["expected_status"]
                and required <= set(sources)
            ),
        })
    except Exception as error:
        record.update({
            "valid": False,
            "correct": False,
            "error": f"{type(error).__name__}: {error}",
        })
    finally:
        record["elapsed_seconds"] = round(
            time.perf_counter() - start, 3
        )
        record["usage"] = captured.get("usage")
        record["finish_reason"] = captured.get("finish_reason")
        record["content"] = captured.get("content")
        record["reasoning"] = captured.get("reasoning")
        record["reasoning_chars"] = len(
            captured.get("reasoning") or ""
        )
        record["visible_chars"] = len(
            captured.get("content") or ""
        )
        record["reasoning_only_truncation"] = (
            record["finish_reason"] == "length"
            and record["visible_chars"] == 0
            and record["reasoning_chars"] > 0
        )

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark semantic RP criterion adjudication"
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--evaluator-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    if not 1 <= args.repetitions <= 100:
        parser.error("repetitions must be 1..100")

    suite_bytes = Path(args.suite).read_bytes()
    config_bytes = Path(args.evaluator_config).read_bytes()

    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    suite = validate_suite(json.loads(suite_bytes))
    config = json.loads(config_bytes)
    from .evaluator_runtime import validate_evaluator_config
    settings = validate_evaluator_config(config)
    if settings is None or settings["mode"] != "llm":
        parser.error("LLM evaluator configuration required")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"output already exists: {output}")

    # Exclusive creation prevents accidentally overwriting results.
    with output.open("x", encoding="utf-8") as file:
        for repetition in range(1, args.repetitions + 1):
            for case in suite["cases"]:
                print(
                    f"{case['id']} | repetition {repetition}",
                    flush=True,
                )
                record = run_case(case, settings)
                record.update({
                    "suite_id": suite["suite_id"],
                    "suite_sha256": suite_sha256,
                    "evaluator_config_sha256": config_sha256,
                    "repetition": repetition,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "model": settings["backend"]["model"],
                    "sampling": settings["sampling"],
                })
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                file.flush()

                print(
                    f"  status={record.get('status')} "
                    f"correct={record['correct']} "
                    f"valid={record['valid']} "
                    f"time={record['elapsed_seconds']}s "
                    f"tokens={(record.get('usage') or {}).get('completion_tokens')}",
                    flush=True,
                )

    print("Results:", output)


if __name__ == "__main__":
    main()
