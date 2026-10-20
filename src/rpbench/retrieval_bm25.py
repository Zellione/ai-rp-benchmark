"""Dependency-free, deterministic BM25 retrieval baseline."""

from __future__ import annotations

import math
from collections import Counter

from .long_context import (
    relevance_terms,
    tokens,
    validate_settings,
)

VERSION = "bm25-retrieval-v1"


def select_context(
    history,
    secret,
    *,
    window_turns=12,
    evidence_turns=8,
    k1=1.5,
    b=0.75,
):
    validate_settings(window_turns, evidence_turns)

    if not (k1 > 0 and 0 <= b <= 1):
        raise ValueError("Invalid BM25 parameters")

    turns = [item["turn"] for item in history]
    if (
        any(type(t) is not int or t < 1 for t in turns)
        or turns != sorted(set(turns))
    ):
        raise ValueError("Invalid history sequence")

    if any(
        not isinstance(item.get("player"), str)
        or not isinstance(item.get("npc"), str)
        for item in history
    ):
        raise ValueError("Invalid history text")

    recent = list(history[-window_turns:])
    older = list(history[:-window_turns])

    query = relevance_terms(secret)

    # Count term frequency, unlike the set-based baseline.
    def words(text):
        import re
        return re.findall(r"\w+", text.casefold())

    documents = [
        Counter(words(item["player"]) + words(item["npc"]))
        for item in older
    ]

    n = len(documents)
    avg_length = (
        sum(sum(doc.values()) for doc in documents) / n
        if n else 1.0
    )

    df = Counter()
    for doc in documents:
        df.update(doc.keys())

    ranked = []
    for item, doc in zip(older, documents):
        length = sum(doc.values())
        score = 0.0

        for term in query:
            freq = doc.get(term, 0)
            if not freq:
                continue

            idf = math.log(
                1 + (n - df[term] + 0.5) / (df[term] + 0.5)
            )

            denominator = (
                freq + k1 * (
                    1 - b + b * length / avg_length
                )
            )

            score += (
                idf * freq * (k1 + 1) / denominator
            )

        if score > 0:
            ranked.append((score, item["turn"], item))

    ranked.sort(key=lambda x: (-x[0], -x[1]))

    selected = sorted(
        (item for _, _, item in ranked[:evidence_turns]),
        key=lambda x: x["turn"],
    )

    covered = len(recent) + len(selected)

    return {
        "retrieval_version": VERSION,
        "recent_context": recent,
        "retrieved_older_evidence": selected,
        "coverage": {
            "total_prior_turns": len(history),
            "included_prior_turns": covered,
            "omitted_prior_turns": len(history) - covered,
            "older_matching_candidates": len(ranked),
            "window_turns": window_turns,
            "evidence_turns": evidence_turns,
            "history_complete": covered == len(history),
            "retrieval_is_exhaustive": covered == len(history),
        },
    }
