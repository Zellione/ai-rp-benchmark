"""Compare retrieval baselines against the same frozen gold cases."""

from __future__ import annotations

import argparse
from functools import partial
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
from .embedding_cache import CachedEmbedder
from .retrieval_dense import (
    embed as dense_embed,
    VERSION as DENSE_VERSION,
    select_context as dense,
)
from .core import load
from .retrieval_benchmark_v2 import (
    cases,
    evaluate_case,
    summarize,
)


def compare(
    output, window_turns=12, evidence_turns=8,
    embedding_config=None,
    embedding_cache=None,
):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)

    dataset = list(cases())
    systems = {
        LEXICAL_VERSION: lexical,
        BM25_VERSION: bm25,
    }

    cache = None
    if embedding_cache is not None and embedding_config is None:
        raise ValueError('Embedding config required for cache')
    if embedding_config is not None:
        cfg = load(embedding_config)
        if embedding_cache is not None:
            cache = CachedEmbedder(
                embedding_cache, cfg, dense_embed
            )
        systems[DENSE_VERSION] = partial(
            dense,
            config=cfg,
            embed_fn=cache if cache is not None else dense_embed,
            batch_size=cfg.get('batch_size', 32),
        )

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
        "embedding_cache": str(embedding_cache)
        if embedding_cache is not None else None,
        "embedding_config": str(embedding_config)
        if embedding_config is not None else None,
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

    if cache is not None:
        summaries['embedding_cache'] = cache.stats()
    dump(root / "summary.json", summaries)
    return root, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--evidence-turns", type=int, default=8)
    parser.add_argument("--embedding-config")
    parser.add_argument("--embedding-cache")
    args = parser.parse_args()

    root, summary = compare(
        args.out, args.window_turns, args.evidence_turns,
        embedding_config=args.embedding_config,
        embedding_cache=args.embedding_cache,
    )
    print(root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
