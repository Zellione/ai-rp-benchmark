"""Deterministic long-context retrieval benchmark.

Synthetic evidence is defined independently of retrieval.
No LLM or inference server is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .core import dump, load
from .long_context import VERSION, select_context


BENCHMARK_VERSION = "retrieval-gold-v1"
LENGTHS = (100, 200, 400)

SECRET = {
    "id": "vault-key-location",
    "owner": "mara-vey",
    "fact": (
        "The vault key is hidden beneath the blue lantern "
        "in Mara's office."
    ),
    "exact_markers": ["blue lantern"],
    "context_markers": [
        "vault", "key", "hidden", "office"
    ],
}


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def base_history(length):
    return [
        {
            "turn": turn,
            "player": (
                f"We continue cataloguing ordinary document "
                f"batch {turn}."
            ),
            "npc": (
                f"Finn sorts accession record {turn} "
                "and closes the ledger."
            ),
        }
        for turn in range(1, length + 1)
    ]


def build_case(length, kind):
    if length not in LENGTHS:
        raise ValueError("Unsupported history length")

    kinds = {
        "explicit_transfer",
        "indirect_transfer",
        "opaque_callback",
        "distractor_competition",
        "no_transfer",
    }
    if kind not in kinds:
        raise ValueError(f"Unknown case kind: {kind}")

    history = base_history(length)
    evidence_turn = 20
    relevant = []

    if kind == "explicit_transfer":
        history[evidence_turn - 1]["npc"] = (
            "Mara privately tells Finn: the vault key is "
            "hidden beneath the blue lantern in my office."
        )
        relevant = [evidence_turn]

    elif kind == "indirect_transfer":
        history[evidence_turn - 1]["npc"] = (
            "Mara whispers to Finn: search below the "
            "sapphire-colored light in my private study. "
            "That is where I concealed the object."
        )
        relevant = [evidence_turn]

    elif kind == "opaque_callback":
        history[evidence_turn - 1]["npc"] = (
            "Mara takes Finn aside, shows him the hiding place "
            "without describing it aloud, and instructs him "
            "to remember what she demonstrated."
        )
        relevant = [evidence_turn]

    elif kind == "distractor_competition":
        history[evidence_turn - 1]["npc"] = (
            "Mara privately tells Finn: the vault key is "
            "hidden beneath the blue lantern in my office."
        )
        relevant = [evidence_turn]
        for turn in range(30, length - 12, 7):
            history[turn - 1]["npc"] = (
                "The archive inventory mentions a vault key "
                "and a blue lantern, but records no transfer "
                f"of knowledge. Inventory entry {turn}."
            )

    else:
        for turn in range(30, length - 12, 13):
            history[turn - 1]["npc"] = (
                "An unrelated catalogue entry refers to "
                "a blue lantern and a vault key."
            )

    # The final NPC turn will be the judge target,
    # never part of the retrievable prior history.
    target = {
        "turn": length + 1,
        "player": "Finn, what do you remember?",
        "npc": "Finn responds to the question.",
    }

    return {
        "id": f"{kind}-{length}",
        "kind": kind,
        "history_length": length,
        "history": history,
        "target": target,
        "secret": dict(SECRET),
        "gold_relevant_turns": relevant,
        "gold_evidence_definition": (
            "The scripted earlier scene establishing "
            "Finn's opportunity to learn the protected fact."
        ),
    }


def cases():
    for length in LENGTHS:
        for kind in (
            "explicit_transfer",
            "indirect_transfer",
            "opaque_callback",
            "distractor_competition",
            "no_transfer",
        ):
            yield build_case(length, kind)


def evaluate_case(case, window_turns, evidence_turns):
    history = case["history"]
    selection = select_context(
        history,
        case["secret"],
        window_turns=window_turns,
        evidence_turns=evidence_turns,
    )

    retrieved = {
        t["turn"]
        for t in (
            selection["recent_context"]
            + selection["retrieved_older_evidence"]
        )
    }
    gold = set(case["gold_relevant_turns"])
    hits = gold & retrieved

    return {
        "case_id": case["id"],
        "kind": case["kind"],
        "history_length": case["history_length"],
        "gold_relevant_turns": sorted(gold),
        "retrieved_relevant_turns": sorted(hits),
        "selected_turns": sorted(retrieved),
        "retrieved_older_turns": [
            t["turn"]
            for t in selection["retrieved_older_evidence"]
        ],
        "false_positive_turns": sorted(retrieved - gold),
        "relevant_total": len(gold),
        "relevant_found": len(hits),
        "recall_at_k": (
            len(hits) / len(gold)
            if gold else None
        ),
        "precision_at_k": (
            len(hits) / len(retrieved)
            if retrieved else None
        ),
        "negative_control": not gold,
        "coverage": selection["coverage"],
    }


def summarize(results):
    positives = [
        result for result in results
        if not result["negative_control"]
    ]
    negatives = [
        result for result in results
        if result["negative_control"]
    ]

    def average(values):
        return sum(values) / len(values) if values else None

    return {
        "total_cases": len(results),
        "positive_cases": len(positives),
        "negative_controls": len(negatives),
        "macro_recall_at_k": average([
            r["recall_at_k"] for r in positives
        ]),
        "macro_precision_at_k_positive_cases": average([
            r["precision_at_k"]
            for r in positives
            if r["precision_at_k"] is not None
        ]),
        "positive_cases_with_full_recall": sum(
            r["recall_at_k"] == 1.0
            for r in positives
        ),
        "negative_control_selected_turns": sum(
            len(r["selected_turns"])
            for r in negatives
        ),
        "results_by_length": {
            str(length): {
                "positive_cases": sum(
                    r["history_length"] == length
                    for r in positives
                ),
                "macro_recall_at_k": average([
                    r["recall_at_k"]
                    for r in positives
                    if r["history_length"] == length
                ]),
            }
            for length in LENGTHS
        },
        "notes": (
            "Lexical retrieval against scripted evidence. "
            "False-positive turn counts include the required "
            "recent context window. This is retrieval quality, "
            "not semantic judgment accuracy."
        ),
    }


def run_benchmark(output, window_turns=12, evidence_turns=8):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    generated = list(cases())
    results = [
        evaluate_case(case, window_turns, evidence_turns)
        for case in generated
    ]

    manifest = {
        "benchmark": BENCHMARK_VERSION,
        "retriever": VERSION,
        "window_turns": window_turns,
        "evidence_turns": evidence_turns,
        "dataset_sha256": stable_hash(generated),
        "case_ids": [case["id"] for case in generated],
        "gold_label_origin": "deterministic scripted cases",
    }

    dump(root / "manifest.json", manifest)

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
    parser = argparse.ArgumentParser(
        description="Offline deterministic retrieval benchmark"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--evidence-turns", type=int, default=8)
    args = parser.parse_args()

    root, summary = run_benchmark(
        args.out,
        window_turns=args.window_turns,
        evidence_turns=args.evidence_turns,
    )
    print(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
