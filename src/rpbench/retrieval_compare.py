"""Compare retrieval baselines against the same frozen gold cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import dump
from .long_context import (
    VERSION as LEXICAL_VERSION,
    select_context as lexical,
)
from .retrieval_bm25 import (
    VERSION as BM25_VERSION,
    select_context as bm25,
)
from .retrieval_benchmark import stable_hash
from .retrieval_benchmark_v2 import (
    cases,
    evaluate_case,
    summarize,
)


def compare(output, window_turns=12, evidence_turns=8):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    dataset = list(cases())
    systems = {
        LEXICAL_VERSION: lexical,
        BM25_VERSION: bm25,
    }

    all_results = {}
    summaries = {}

    for name, retriever in systems.items():
        results = [
            evaluate_case(
                case, window_turns, evidence_turns,
                retriever=retriever,
            )
            for case in dataset
        ]
        all_results[name] = results
        summaries[name] = summarize(results)

    dump(root / "manifest.json", {
        "benchmark": "retrieval-comparison-v1",
        "dataset_sha256": stable_hash(dataset),
        "retrievers": list(systems),
        "window_turns": window_turns,
        "evidence_turns": evidence_turns,
    })

    with (root / "case_results.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for name, results in all_results.items():
            for result in results:
                f.write(json.dumps({
                    "retriever": name,
                    **result,
                }, ensure_ascii=False) + "\n")

    dump(root / "summary.json", summaries)
    return root, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--evidence-turns", type=int, default=8)
    args = parser.parse_args()

    root, summary = compare(
        args.out, args.window_turns, args.evidence_turns
    )
    print(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
