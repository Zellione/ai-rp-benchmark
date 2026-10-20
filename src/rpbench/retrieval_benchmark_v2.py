"""Adversarial long-context retrieval benchmark v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import dump
from .long_context import VERSION, select_context
from .retrieval_benchmark import (
    SECRET,
    base_history,
    stable_hash,
)

VERSION_BENCH = "retrieval-gold-v2"
LENGTHS = (100, 200, 400)

KINDS = (
    "literal_transfer",
    "semantic_transfer",
    "opaque_reference",
    "distractor_saturation",
    "multi_hop",
    "negative_control",
)


def make_case(length, kind):
    if length not in LENGTHS or kind not in KINDS:
        raise ValueError("Unsupported benchmark case")

    history = base_history(length)
    first = length // 5
    second = length // 2
    gold = []

    if kind == "literal_transfer":
        history[first - 1]["npc"] = (
            "Mara privately tells Finn that the vault key "
            "is beneath the blue lantern in her office."
        )
        gold = [first]

    elif kind == "semantic_transfer":
        history[first - 1]["npc"] = (
            "The curator takes the apprentice aside and "
            "quietly explains where the concealed object "
            "can be recovered."
        )
        gold = [first]

    elif kind == "opaque_reference":
        history[first - 1]["npc"] = (
            "She unfolds a private diagram for him, points "
            "at one marked spot, and makes him memorize it."
        )
        gold = [first]

    elif kind == "distractor_saturation":
        history[first - 1]["npc"] = (
            "Mara privately tells Finn the vault key "
            "is beneath the blue lantern in her office."
        )
        gold = [first]

        for turn in range(first + 2, length - 12, 3):
            history[turn - 1]["npc"] = (
                "The vault key and blue lantern are listed "
                "together in an unrelated catalogue. "
                f"Entry number {turn}."
            )

    elif kind == "multi_hop":
        history[first - 1]["npc"] = (
            "The curator shows the apprentice a folded map "
            "and asks him to memorize the marked location."
        )
        history[second - 1]["npc"] = (
            "Later, he passes that private information "
            "to the visiting researcher."
        )
        gold = [first, second]

    else:
        for turn in range(first, length - 12, 9):
            history[turn - 1]["npc"] = (
                "A catalogue lists an unrelated vault key "
                "and blue lantern among exhibition props."
            )

    return {
        "id": f"{kind}-{length}",
        "kind": kind,
        "history_length": length,
        "history": history,
        "secret": dict(SECRET),
        "target": {
            "turn": length + 1,
            "player": "What information was passed along?",
            "npc": "The characters continue their discussion.",
        },
        "gold_relevant_turns": gold,
        "annotation_note": (
            "Scripted provenance references. Semantic cases "
            "are retrieval stress tests, not proof that the "
            "target character learned the complete secret."
        ),
    }


def cases():
    for length in LENGTHS:
        for kind in KINDS:
            yield make_case(length, kind)


def evaluate_case(case, window_turns=12, evidence_turns=8):
    selection = select_context(
        case["history"],
        case["secret"],
        window_turns=window_turns,
        evidence_turns=evidence_turns,
    )

    recent = {
        item["turn"] for item in selection["recent_context"]
    }
    older = {
        item["turn"]
        for item in selection["retrieved_older_evidence"]
    }
    selected = recent | older
    gold = set(case["gold_relevant_turns"])

    return {
        "case_id": case["id"],
        "kind": case["kind"],
        "history_length": case["history_length"],
        "gold_relevant_turns": sorted(gold),
        "retrieved_relevant_turns": sorted(selected & gold),
        "older_relevant_turns": sorted(older & gold),
        "older_selected_turns": sorted(older),
        "missing_relevant_turns": sorted(gold - selected),
        "recall_at_k": (
            len(selected & gold) / len(gold)
            if gold else None
        ),
        "older_precision_at_k": (
            len(older & gold) / len(older)
            if older else None
        ),
        "negative_control": not gold,
        "coverage": selection["coverage"],
    }


def summarize(results):
    positive = [
        result for result in results
        if not result["negative_control"]
    ]

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "total_cases": len(results),
        "positive_cases": len(positive),
        "negative_controls": len(results) - len(positive),
        "macro_recall_at_k": mean([
            r["recall_at_k"] for r in positive
        ]),
        "older_precision_at_k": mean([
            r["older_precision_at_k"] for r in positive
            if r["older_precision_at_k"] is not None
        ]),
        "complete_recall_cases": sum(
            r["recall_at_k"] == 1
            for r in positive
        ),
        "by_kind": {
            kind: {
                "macro_recall_at_k": mean([
                    r["recall_at_k"]
                    for r in positive
                    if r["kind"] == kind
                ])
            }
            for kind in KINDS
            if kind != "negative_control"
        },
    }


def run_benchmark(output, window_turns=12, evidence_turns=8):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    dataset = list(cases())
    results = [
        evaluate_case(case, window_turns, evidence_turns)
        for case in dataset
    ]

    dump(root / "manifest.json", {
        "benchmark": VERSION_BENCH,
        "retriever": VERSION,
        "dataset_sha256": stable_hash(dataset),
        "window_turns": window_turns,
        "evidence_turns": evidence_turns,
    })

    with (root / "case_results.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for result in results:
            f.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )

    summary = summarize(results)
    dump(root / "summary.json", summary)
    return root, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--evidence-turns", type=int, default=8)
    args = parser.parse_args()

    root, summary = run_benchmark(
        args.out, args.window_turns, args.evidence_turns
    )
    print(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
