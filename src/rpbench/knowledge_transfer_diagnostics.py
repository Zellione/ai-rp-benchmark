"""Field-level diagnostics for knowledge transfer validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .knowledge_transfer_validation import normalize_fact


def event_pair_key(event):
    return (
        event["source_character_id"],
        event["recipient_character_id"],
    )


def match_case(gold, predicted):
    candidates = []

    for gi, expected in enumerate(gold):
        for pi, actual in enumerate(predicted):
            if event_pair_key(expected) != event_pair_key(actual):
                continue

            same_kind = expected["kind"] == actual["kind"]
            same_prior = (
                expected["recipient_prior_knowledge"]
                == actual["recipient_prior_knowledge"]
            )
            same_fact = (
                normalize_fact(expected["fact"])
                == normalize_fact(actual["fact"])
            )

            quality = (
                int(same_kind) + int(same_prior) + int(same_fact)
            )
            candidates.append(
                (quality, gi, pi, same_kind, same_prior, same_fact)
            )

    candidates.sort(
        key=lambda row: (-row[0], row[1], row[2])
    )

    used_gold = set()
    used_pred = set()
    matches = []

    for _, gi, pi, kind, prior, fact in candidates:
        if gi in used_gold or pi in used_pred:
            continue

        used_gold.add(gi)
        used_pred.add(pi)

        matches.append({
            "expected_index": gi,
            "predicted_index": pi,
            "kind_match": kind,
            "prior_knowledge_match": prior,
            "fact_exact_match": fact,
            "expected_kind": gold[gi]["kind"],
            "predicted_kind": predicted[pi]["kind"],
            "expected_fact": gold[gi]["fact"],
            "predicted_fact": predicted[pi]["fact"],
        })

    return {
        "matches": matches,
        "unmatched_gold": [
            gold[i] for i in range(len(gold))
            if i not in used_gold
        ],
        "unmatched_predicted": [
            predicted[i] for i in range(len(predicted))
            if i not in used_pred
        ],
    }


def analyze(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    cases = []
    for row in rows:
        cases.append({
            "case_id": row["case_id"],
            **match_case(row["expected"], row["predicted"]),
        })

    matches = [
        m for case in cases for m in case["matches"]
    ]

    def accuracy(field):
        return (
            sum(m[field] for m in matches) / len(matches)
            if matches else None
        )

    return {
        "cases": len(cases),
        "pair_aligned_events": len(matches),
        "unmatched_gold": sum(
            len(c["unmatched_gold"]) for c in cases
        ),
        "unmatched_predictions": sum(
            len(c["unmatched_predicted"]) for c in cases
        ),
        "kind_accuracy_on_aligned": accuracy("kind_match"),
        "prior_accuracy_on_aligned": accuracy(
            "prior_knowledge_match"
        ),
        "fact_exact_accuracy_on_aligned": accuracy(
            "fact_exact_match"
        ),
        "note": (
            "Alignment requires matching source and recipient. "
            "All accuracies are conditional on aligned events."
        ),
    }, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    summary, cases = analyze(args.results)
    print(json.dumps(
        summary, indent=2, ensure_ascii=False
    ))

    for case in cases:
        print(f"\n=== {case['case_id']} ===")
        print("Aligned:", len(case["matches"]))
        for match in case["matches"]:
            if not (
                match["kind_match"]
                and match["prior_knowledge_match"]
                and match["fact_exact_match"]
            ):
                print(json.dumps(
                    match, indent=2, ensure_ascii=False
                ))
        if case["unmatched_gold"]:
            print("Missed:", case["unmatched_gold"])
        if case["unmatched_predicted"]:
            print("Extra:", case["unmatched_predicted"])


if __name__ == "__main__":
    main()
