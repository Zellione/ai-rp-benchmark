"""Independent diagnostics for attribution gold-label outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLES = ("speaker_id", "actor_id", "subject_id")


def overlap(a, b):
    left = max(a["start_offset"], b["start_offset"])
    right = min(a["end_offset"], b["end_offset"])
    return max(0, right - left)


def span_iou(a, b):
    intersection = overlap(a, b)
    union = (
        max(a["end_offset"], b["end_offset"])
        - min(a["start_offset"], b["start_offset"])
    )
    return intersection / union if union else 0.0


def compare_pair(gold, predicted):
    return {
        "span_iou": span_iou(gold, predicted),
        "kind_match": gold["kind"] == predicted["kind"],
        "source_match": (
            gold["narrative_source"]
            == predicted["narrative_source"]
        ),
        "role_matches": {
            role: gold[role] == predicted[role]
            for role in ROLES
        },
        "exact_match": all(
            gold[key] == predicted[key]
            for key in (
                "start_offset", "end_offset", "kind",
                "speaker_id", "actor_id", "subject_id",
                "narrative_source",
            )
        ),
    }


def align_events(gold, predicted, threshold=0.5):
    """Greedy one-to-one matching based only on span overlap.

    Matching does not use kind or character labels, so
    category errors remain observable instead of disappearing.
    """
    candidates = []
    for i, g in enumerate(gold):
        for j, p in enumerate(predicted):
            iou = span_iou(g, p)
            if iou >= threshold:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_gold = set()
    used_pred = set()
    aligned = []

    for _, i, j in candidates:
        if i in used_gold or j in used_pred:
            continue
        used_gold.add(i)
        used_pred.add(j)
        aligned.append(compare_pair(gold[i], predicted[j]))

    return {
        "pairs": aligned,
        "unmatched_gold": len(gold) - len(used_gold),
        "unmatched_predicted": len(predicted) - len(used_pred),
    }


def analyze(path, threshold=0.5):
    results = []
    for line in Path(path).read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            results.append(json.loads(line))

    diagnostics = []
    for result in results:
        aligned = align_events(
            result["expected"],
            result["predicted"],
            threshold,
        )
        diagnostics.append({
            "case_id": result["case_id"],
            **aligned,
        })

    pairs = [
        pair
        for case in diagnostics
        for pair in case["pairs"]
    ]
    n = len(pairs)

    summary = {
        "cases": len(diagnostics),
        "aligned_events": n,
        "unmatched_gold": sum(
            c["unmatched_gold"] for c in diagnostics
        ),
        "unmatched_predictions": sum(
            c["unmatched_predicted"] for c in diagnostics
        ),
        "mean_span_iou": (
            sum(p["span_iou"] for p in pairs) / n
            if n else None
        ),
        "kind_accuracy_on_aligned": (
            sum(p["kind_match"] for p in pairs) / n
            if n else None
        ),
        "source_accuracy_on_aligned": (
            sum(p["source_match"] for p in pairs) / n
            if n else None
        ),
        "role_accuracy_on_aligned": {
            role: (
                sum(p["role_matches"][role] for p in pairs) / n
                if n else None
            )
            for role in ROLES
        },
        "note": (
            "Role and category accuracies are conditional "
            "on span-aligned events, not full retrieval recall."
        ),
    }
    return summary, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    summary, cases = analyze(
        args.results, args.threshold
    )
    print(json.dumps(
        summary, indent=2, ensure_ascii=False
    ))

    print("\n=== PER-CASE DIAGNOSTICS ===")
    for case in cases:
        print(
            case["case_id"],
            "aligned=", len(case["pairs"]),
            "missed=", case["unmatched_gold"],
            "extra=", case["unmatched_predicted"],
        )


if __name__ == "__main__":
    main()
