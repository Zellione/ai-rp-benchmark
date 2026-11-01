"""Versioned, resumable offline-judge calibration."""

import argparse
import json
import os
from pathlib import Path

from .core import Backend, dump, load
from .evaluator_llm import parse_evaluator_json
from .offline_goal_audit import (
    INSTRUCTIONS,
    SCHEMA_VERSION,
    encode,
    normalize_evidence_refs,
    sha256_bytes,
    validate_verdict,
)


DEFAULT_BASE_URL = "https://nano-gpt.com/api/subscription"


def load_suite(fixture_path, configuration_path):
    suite = load(Path(fixture_path))
    config = load(Path(configuration_path))
    definition = config["runtime"]["goal_criteria"]

    goal_id = suite["goal_id"]
    criteria = definition["goals"][goal_id]["criteria"]
    criterion_ids = [item["id"] for item in criteria]

    if len(criterion_ids) != len(set(criterion_ids)):
        raise ValueError("Duplicate criterion IDs")

    cases = suite["cases"]
    case_ids = [case["id"] for case in cases]

    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("Missing or duplicate calibration cases")

    for case in cases:
        if not case["observations"]:
            raise ValueError("Empty calibration transcript")

        expected = case["expected"]

        if set(expected) != set(criterion_ids):
            raise ValueError("Calibration criterion mismatch")

        reference = {
            "schema_version": SCHEMA_VERSION,
            "goal_id": goal_id,
            "criteria": [
                {
                    "criterion_id": cid,
                    **expected[cid],
                }
                for cid in criterion_ids
            ],
        }

        validate_verdict(
            reference, goal_id, definition, case["observations"]
        )

    return suite, definition


def make_context(suite, definition, case):
    goal_id = suite["goal_id"]

    return {
        "goal_id": goal_id,
        "criteria": [
            {
                "criterion_id": item["id"],
                "description": item["description"],
            }
            for item in definition["goals"][goal_id]["criteria"]
        ],
        "transcript": case["observations"],
    }


def make_identity(suite, definition, case, model, base_url, max_tokens):
    return {
        "calibration_schema_version": 1,
        "audit_schema_version": SCHEMA_VERSION,
        "case_id": case["id"],
        "case_sha256": sha256_bytes(encode(case)),
        "goal_id": suite["goal_id"],
        "definition_sha256": sha256_bytes(encode(definition)),
        "instructions_sha256": sha256_bytes(
            INSTRUCTIONS.encode("utf-8")
        ),
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
    }


def compare(case, verdict):
    expected = case["expected"]
    rows = []

    for item in verdict["criteria"]:
        cid = item["criterion_id"]
        reference = expected[cid]

        status_match = item["status"] == reference["status"]

        turn_match = (
            status_match
            and item["first_satisfied_turn"]
            == reference["first_satisfied_turn"]
        )

        rows.append({
            "criterion_id": cid,
            "expected_status": reference["status"],
            "actual_status": item["status"],
            "status_match": status_match,
            "expected_turn": reference["first_satisfied_turn"],
            "actual_turn": item["first_satisfied_turn"],
            "turn_match": turn_match,
        })

    return rows


def calibrate_case(
    suite, definition, case, output, model,
    base_url, max_tokens, execute=False,
):
    output = Path(output)
    identity = make_identity(
        suite, definition, case, model, base_url, max_tokens
    )
    identity_hash = sha256_bytes(encode(identity))

    context = make_context(suite, definition, case)

    if not execute:
        return {
            "case_id": case["id"],
            "status": "dry_run",
            "identity_sha256": identity_hash,
            "turns": max(
                item["turn"] for item in case["observations"]
            ),
        }

    identity_file = output / "identity.json"
    raw_file = output / "raw_response.json"
    result_file = output / "result.json"

    if identity_file.exists():
        if load(identity_file) != identity:
            raise ValueError("Calibration identity mismatch")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty uninitialized calibration output")

    if result_file.exists():
        existing = load(result_file)

        if existing.get("identity_sha256") != identity_hash:
            raise ValueError("Result identity mismatch")

        if existing.get("status") == "valid":
            return existing

        # Revalidate cached invalid results using the original raw answer.
        # Never issue another API request when a raw response exists.
        if not raw_file.exists():
            raise ValueError(
                "Invalid cached result has no raw response"
            )

    if raw_file.exists():
        raw = load(raw_file)

        if raw.get("identity_sha256") != identity_hash:
            raise ValueError("Raw calibration identity mismatch")

        print("Reusing raw response:", case["id"])

    else:
        api_key = os.getenv("NANOGPT_API_KEY")
        if not api_key:
            raise ValueError("Missing NANOGPT_API_KEY")

        backend = Backend(
            name="nanogpt",
            base_url=base_url,
            model=model,
            mode="chat",
            api_key=api_key,
            timeout=300,
        )

        url, payload = backend.prepare(
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": json.dumps(
                        context, ensure_ascii=False
                    ),
                },
            ],
            {"temperature": 0, "max_tokens": max_tokens},
            {},
        )

        content, response = backend.generate(url, payload)

        if not isinstance(content, str):
            raise ValueError("Invalid response content")

        choice = (response.get("choices") or [{}])[0]

        raw = {
            "identity_sha256": identity_hash,
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
        }

        output.mkdir(parents=True, exist_ok=True)
        dump(identity_file, identity)
        dump(raw_file, raw)
        print("Saved raw response:", case["id"])

    result = {
        "case_id": case["id"],
        "identity_sha256": identity_hash,
        "expected_criterion_ids": [
            item["id"]
            for item in definition["goals"][suite["goal_id"]]["criteria"]
        ],
        "finish_reason": raw.get("finish_reason"),
        "usage": raw.get("usage"),
        "status": "invalid",
        "error": None,
        "normalizations": [],
        "verdict": None,
        "comparison": [],
    }

    try:
        if raw.get("finish_reason") == "length":
            raise ValueError("Judge exhausted output budget")

        proposal = parse_evaluator_json(raw["content"])

        proposal, changes = normalize_evidence_refs(
            proposal, case["observations"]
        )
        result["normalizations"] = changes

        verdict = validate_verdict(
            proposal,
            suite["goal_id"],
            definition,
            case["observations"],
        )

        result["verdict"] = verdict
        result["comparison"] = compare(case, verdict)
        result["status"] = "valid"

    except (ValueError, TypeError, KeyError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    dump(result_file, result)
    return result


def summarize(results):
    """Report conditional and end-to-end calibration accuracy."""

    valid_results = [
        result for result in results
        if result["status"] == "valid"
    ]

    rows = [
        row
        for result in valid_results
        for row in result["comparison"]
    ]

    satisfied_rows = [
        row for row in rows
        if row["expected_status"] == "satisfied"
    ]

    criteria_total = sum(
        len(result["comparison"])
        if result["status"] == "valid"
        else len(result.get("expected_criterion_ids", []))
        for result in results
    )

    status_correct = sum(
        row["status_match"] for row in rows
    )

    end_to_end_correct = sum(
        row["status_match"]
        and (
            row["expected_status"] != "satisfied"
            or row["turn_match"]
        )
        for row in rows
    )

    perfect_cases = sum(
        bool(result["comparison"])
        and all(
            row["status_match"]
            and (
                row["expected_status"] != "satisfied"
                or row["turn_match"]
            )
            for row in result["comparison"]
        )
        for result in valid_results
    )

    summary = {
        "cases": len(results),
        "valid_cases": len(valid_results),
        "invalid_cases": len(results) - len(valid_results),
        "criteria_scored": len(rows),
        "criteria_total": criteria_total,
        "status_correct": status_correct,
        "status_accuracy": (
            status_correct / len(rows)
            if rows else None
        ),
        "satisfied_turn_correct": sum(
            row["turn_match"] for row in satisfied_rows
        ),
        "satisfied_turn_total": len(satisfied_rows),
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": (
            end_to_end_correct / criteria_total
            if criteria_total else None
        ),
        "perfect_cases": perfect_cases,
        "perfect_case_rate": (
            perfect_cases / len(results)
            if results else None
        ),
        "normalizations": sum(
            len(result["normalizations"])
            for result in results
        ),
        "results": [
            {
                "case_id": result["case_id"],
                "status": result["status"],
                "error": result["error"],
                "comparison": result["comparison"],
                "normalizations": result["normalizations"],
            }
            for result in results
        ],
    }

    return summary


def run_suite(
    fixture, configuration, output, model,
    base_url=DEFAULT_BASE_URL, max_tokens=16384,
    execute=False,
):
    suite, definition = load_suite(fixture, configuration)
    output = Path(output).resolve()

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    if execute:
        for source in (fixture, configuration):
            source = Path(source).resolve()
            if output == source.parent or source.parent in output.parents:
                raise ValueError(
                    "Calibration output must not be inside source directory"
                )

    results = []

    for case in suite["cases"]:
        result = calibrate_case(
            suite, definition, case,
            output / case["id"], model,
            base_url, max_tokens, execute=execute,
        )
        results.append(result)
        print(case["id"], result["status"])

    if not execute:
        return results

    summary = summarize(results)
    dump(output / "summary.json", summary)

    print(json.dumps({
        key: value
        for key, value in summary.items() if key != "results"
    }, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--execute", action="store_true",
        help="Send requests; default is dry-run",
    )

    args = parser.parse_args()

    run_suite(
        args.fixture,
        args.definition,
        args.out,
        args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
