"""Offline gold-label validation for knowledge transfer extraction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .core import append, dump, load
from .judge import request_chat
from .knowledge_transfers import (
    KINDS,
    make_payload,
    parse_response,
)

PRIOR = {"known", "unknown", "not_known"}


def normalize_fact(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def event_key(event):
    return (
        event["kind"],
        event["source_character_id"],
        event["recipient_character_id"],
        event["recipient_prior_knowledge"],
        normalize_fact(event["fact"]),
    )


def validate_dataset(data):
    if data.get("schema_version") != 1:
        raise ValueError("Invalid dataset version")

    characters = data.get("characters")
    cases = data.get("cases")
    if not isinstance(characters, list) or not characters:
        raise ValueError("Missing characters")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Missing cases")

    ids = set()
    for character in characters:
        cid = character.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("Invalid character catalogue")
        ids.add(cid)

    seen = set()
    for case in cases:
        cid = case.get("id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Duplicate or invalid case ID")
        seen.add(cid)
        if not isinstance(case.get("text"), str):
            raise ValueError("Missing case text")
        expected = case.get("expected")
        if not isinstance(expected, list):
            raise ValueError("Missing expected transfers")

        keys = set()
        for event in expected:
            if event.get("kind") not in KINDS:
                raise ValueError("Invalid gold transfer kind")
            for role in (
                "source_character_id",
                "recipient_character_id",
            ):
                value = event.get(role)
                if value is not None and value not in ids:
                    raise ValueError("Unknown gold character")
            if (
                event.get("recipient_prior_knowledge")
                not in PRIOR
            ):
                raise ValueError("Invalid gold prior knowledge")
            if not isinstance(event.get("fact"), str):
                raise ValueError("Invalid gold fact")
            key = event_key(event)
            if key in keys:
                raise ValueError("Duplicate gold transfer")
            keys.add(key)

    return ids


def score(expected, predicted):
    gold = {event_key(e) for e in expected}
    actual = {event_key(e) for e in predicted}
    return {
        "tp": len(gold & actual),
        "fp": len(actual - gold),
        "fn": len(gold - actual),
        "exact_match": gold == actual,
    }


def metrics(results):
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "cases": len(results),
        "exact_case_matches": sum(
            r["exact_match"] for r in results
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0
            else None
        ),
    }


def run_validation(dataset_path, config_path, output):
    data = load(dataset_path)
    config = load(config_path)
    validate_dataset(data)

    if config["backend"]["protocol"] != "openai-chat":
        raise ValueError("Expected OpenAI chat backend")

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    dump(root / "manifest.json", {
        "dataset": data,
        "config": config,
        "status": "running",
        "note": "Candidate extraction; not confirmed knowledge",
    })

    results = []
    completed = False
    try:
        for case in data["cases"]:
            job = {"turn": 1, "text": case["text"]}
            payload = make_payload(
                job, data["characters"], config
            )

            append(root / "judge_events.jsonl", {
                "type": "request",
                "case_id": case["id"],
                "payload": payload,
            })

            try:
                raw = request_chat(config, payload)
                append(root / "judge_events.jsonl", {
                    "type": "raw_response",
                    "case_id": case["id"],
                    "response": raw,
                })
                predicted = parse_response(
                    raw, job, data["characters"]
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
                "expected": case["expected"],
                "predicted": predicted,
                **score(case["expected"], predicted),
                "usage": raw.get("usage"),
            }
            append(root / "case_results.jsonl", result)
            results.append(result)

        summary = {"status": "complete", **metrics(results)}
        dump(root / "summary.json", summary)
        completed = True
        return summary
    finally:
        manifest_path = root / "manifest.json"
        manifest = load(manifest_path)
        manifest["status"] = (
            "complete" if completed else "incomplete"
        )
        dump(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(json.dumps(
        run_validation(args.dataset, args.config, args.out),
        indent=2,
    ))


if __name__ == "__main__":
    main()
