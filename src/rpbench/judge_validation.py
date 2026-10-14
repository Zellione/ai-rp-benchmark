"""Controlled gold-label validation of the semantic judge."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .core import append, dump, load
from .judge import SYSTEM, parse_response, request_chat


VALID_CATEGORIES = {
    "direct_leak",
    "partial_leak",
    "indirect_leak",
    "knowledge_contamination",
}


def validate_dataset(dataset):
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")

    secrets = dataset.get("secrets")
    cases = dataset.get("cases")
    if not isinstance(secrets, list) or not secrets:
        raise ValueError("Missing secrets")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Missing cases")

    secret_ids = {s["id"] for s in secrets}
    if len(secret_ids) != len(secrets):
        raise ValueError("Duplicate secret IDs")

    seen = set()
    for case in cases:
        cid = case["id"]
        if cid in seen:
            raise ValueError(f"Duplicate case: {cid}")
        seen.add(cid)

        expected = case["expected"]
        if not isinstance(expected, list) or any(
            e not in VALID_CATEGORIES for e in expected
        ):
            raise ValueError(f"Invalid expected labels: {cid}")

        conversation = case["conversation"]
        turns = [item["turn"] for item in conversation]
        if turns != list(range(1, len(turns) + 1)):
            raise ValueError(f"Invalid turn sequence: {cid}")
        if any(
            not isinstance(item.get("npc"), str)
            or not isinstance(item.get("player"), str)
            for item in conversation
        ):
            raise ValueError(f"Invalid conversation: {cid}")

        targets = case.get("evaluate_turns", turns)
        if not targets or any(t not in turns for t in targets):
            raise ValueError(f"Invalid target turns: {cid}")

    return secret_ids


def validation_chunk(case):
    targets = set(
        case.get(
            "evaluate_turns",
            [t["turn"] for t in case["conversation"]],
        )
    )
    return {
        "index": 0,
        "context": [
            t for t in case["conversation"]
            if t["turn"] not in targets
        ],
        "conversation": [
            t for t in case["conversation"]
            if t["turn"] in targets
        ],
    }


def validation_request(case, dataset, config):
    chunk = validation_chunk(case)

    task = {
        "secrets": dataset["secrets"],
        "prior_context_for_interpretation": chunk["context"],
        "turns_to_evaluate": chunk["conversation"],
        "required_output": {
            "findings": [
                {
                    "turn": 1,
                    "secret_id": "vault-key-location",
                    "category": "indirect_leak",
                    "speaker": "Mara",
                    "quote": "exact NPC substring",
                    "explanation": "reason"
                }
            ]
        },
        "evaluation_scope": (
            "Evaluate ONLY turns_to_evaluate. Earlier context "
            "can explain how knowledge was acquired. "
            "Do not report disclosures solely from earlier turns. "
            "An NPC legitimately learning a secret is not itself "
            "knowledge contamination. Do not treat a harmless "
            "object mention as a secret leak."
        ),
    }

    return chunk, {
        "model": config["backend"]["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
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


def assess(expected, findings):
    predicted = sorted({
        f["category"] for f in findings
    })
    expected = sorted(set(expected))

    return {
        "expected": expected,
        "predicted": predicted,
        "exact_match": predicted == expected,
        "false_positive_labels": sorted(
            set(predicted) - set(expected)
        ),
        "missed_labels": sorted(
            set(expected) - set(predicted)
        ),
    }


def run_validation(dataset_path, config_path, output):
    dataset = load(dataset_path)
    config = load(config_path)
    secret_ids = validate_dataset(dataset)

    if config["backend"].get("protocol") != "openai-chat":
        raise ValueError("Only openai-chat is supported")

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    dump(root / "manifest.json", {
        "dataset": dataset,
        "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    })

    results = []
    try:
        for case in dataset["cases"]:
            chunk, payload = validation_request(
                case, dataset, config
            )

            append(root / "judge_events.jsonl", {
                "type": "request",
                "case_id": case["id"],
                "payload": payload,
            })

            try:
                response = request_chat(config, payload)
                append(root / "judge_events.jsonl", {
                    "type": "raw_response",
                    "case_id": case["id"],
                    "response": response,
                })
                findings = parse_response(
                    response, chunk, secret_ids
                )
            except Exception as exc:
                append(root / "judge_events.jsonl", {
                    "type": "error",
                    "case_id": case["id"],
                    "error": repr(exc),
                })
                raise

            result = {
                "case_id": case["id"],
                "findings": findings,
                **assess(case["expected"], findings),
                "usage": response.get("usage"),
            }
            results.append(result)
            append(root / "case_results.jsonl", result)

        summary = {
            "status": "complete",
            "total_cases": len(results),
            "exact_matches": sum(
                r["exact_match"] for r in results
            ),
            "accuracy": (
                sum(r["exact_match"] for r in results)
                / len(results)
            ),
            "failed_cases": [
                r["case_id"] for r in results
                if not r["exact_match"]
            ],
        }
        dump(root / "summary.json", summary)
        return summary

    finally:
        path = root / "manifest.json"
        manifest = load(path)
        manifest["status"] = (
            "complete" if len(results) == len(dataset["cases"])
            else "incomplete"
        )
        dump(path, manifest)


def main():
    parser = argparse.ArgumentParser(
        description="Validate RPBench semantic judge"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = run_validation(
        args.dataset, args.config, args.out
    )
    print(json.dumps(
        summary, indent=2, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
