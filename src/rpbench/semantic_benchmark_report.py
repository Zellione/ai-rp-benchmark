"""Reproducible quality report for archived semantic evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

from .semantic_benchmark import validate_suite


def report(suite_file, results_file, *, allow_legacy=False):
    suite_bytes = Path(suite_file).read_bytes()
    suite = validate_suite(json.loads(suite_bytes))
    digest = hashlib.sha256(suite_bytes).hexdigest()

    expected = {
        case["id"]: case
        for case in suite["cases"]
    }

    records = [
        json.loads(line)
        for line in Path(results_file).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not records:
        raise ValueError("Empty results")

    for item in records:
        archived_hash = item.get("suite_sha256")
        if archived_hash is None:
            if not allow_legacy:
                raise ValueError(
                    "Missing suite fingerprint; "
                    "use --allow-legacy only when the "
                    "historical suite is known"
                )
        elif archived_hash != digest:
            raise ValueError("Benchmark suite fingerprint mismatch")

        if item.get("case") not in expected:
            raise ValueError("Unknown benchmark case")
        if (
            "suite_id" in item
            and item["suite_id"] != suite["suite_id"]
        ):
            raise ValueError("Benchmark suite ID mismatch")

    valid = 0
    status_correct = 0
    completion_correct = 0
    false_completions = 0
    missed_completions = 0
    reasoning_truncations = 0
    completion_tokens = 0
    elapsed_seconds = 0.0

    per_case = []

    for item in records:
        target = expected[item["case"]]["expected_status"]
        predicted = item.get("status")
        is_valid = item.get("valid") is True

        status_ok = is_valid and predicted == target
        completion_ok = (
            is_valid
            and (predicted == "satisfied")
            == (target == "satisfied")
        )

        false_completion = (
            is_valid
            and predicted == "satisfied"
            and target != "satisfied"
        )
        missed_completion = (
            is_valid
            and predicted != "satisfied"
            and target == "satisfied"
        )

        required = set(
            expected[item["case"]].get(
                "required_source_turns", []
            )
        )
        cited = item.get("source_turns") or []
        source_ok = is_valid and required <= set(cited)

        if (
            not source_ok
            and required
            and status_ok
        ):
            status_ok = False

        valid += is_valid
        status_correct += status_ok
        completion_correct += completion_ok and source_ok
        false_completions += false_completion
        missed_completions += missed_completion

        reasoning_truncations += (
            item.get("reasoning_only_truncation") is True
        )
        completion_tokens += (
            (item.get("usage") or {}).get(
                "completion_tokens", 0
            ) or 0
        )
        elapsed_seconds += item.get("elapsed_seconds", 0) or 0

        per_case.append({
            "case": item["case"],
            "repetition": item.get("repetition", 1),
            "expected": target,
            "predicted": predicted,
            "valid": is_valid,
            "status_correct": bool(status_ok),
            "completion_correct": bool(
                completion_ok and source_ok
            ),
            "source_correct": source_ok,
        })

    return {
        "suite_id": suite["suite_id"],
        "suite_sha256": digest,
        "legacy_results": any(
            item.get("suite_sha256") is None
            for item in records
        ),
        "evaluations": len(records),
        "valid": valid,
        "status_correct": status_correct,
        "completion_correct": completion_correct,
        "false_completions": false_completions,
        "missed_completions": missed_completions,
        "reasoning_only_truncations": reasoning_truncations,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "cases": per_case,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize archived semantic benchmark results"
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()

    result = report(
        args.suite,
        args.results,
        allow_legacy=args.allow_legacy,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
