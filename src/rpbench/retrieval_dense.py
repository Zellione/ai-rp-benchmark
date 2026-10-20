"""OpenAI-compatible dense embedding retrieval baseline."""

from __future__ import annotations

import json
import math
import os
import urllib.request

from .long_context import validate_settings

VERSION = "dense-nomic-v1"


def embed(texts, config, *, prefix):
    if prefix not in {"search_query", "search_document"}:
        raise ValueError("Invalid embedding prefix")
    if not texts:
        return []

    backend = config["backend"]
    url = backend["base_url"].rstrip("/") + "/v1/embeddings"

    payload = {
        "model": backend["model"],
        "input": [
            f"{prefix}: {text}" for text in texts
        ],
    }

    headers = {"Content-Type": "application/json"}
    key = os.getenv(
        backend.get("api_key_env", "RPBENCH_EMBED_API_KEY")
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=backend.get("timeout", 120)
    ) as response:
        result = json.load(response)

    items = result["data"]
    if len(items) != len(texts):
        raise ValueError("Embedding response count mismatch")

    vectors = [None] * len(texts)
    for item in items:
        index = item["index"]
        vector = item["embedding"]

        if (
            type(index) is not int
            or not 0 <= index < len(texts)
            or vectors[index] is not None
            or not isinstance(vector, list)
            or not vector
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("Invalid embedding response")

        vectors[index] = vector

    if any(vector is None for vector in vectors):
        raise ValueError("Missing embedding")

    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1:
        raise ValueError("Inconsistent embedding dimensions")

    return vectors


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or invalid embedding norm")
    return [value / norm for value in vector]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ")
    return sum(x * y for x, y in zip(a, b))


def document_text(item):
    return (
        f"Player: {item['player']}\n"
        f"NPC: {item['npc']}"
    )


def query_text(secret):
    return (
        "Find earlier conversations showing how a character "
        "learned, shared, or revealed this secret: "
        + secret["fact"]
    )


def select_context(
    history,
    secret,
    *,
    window_turns=12,
    evidence_turns=8,
    config,
    embed_fn=embed,
    batch_size=32,
):
    validate_settings(window_turns, evidence_turns)

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid embedding batch size")

    recent = list(history[-window_turns:])
    older = list(history[:-window_turns])

    if not older or evidence_turns == 0:
        selected = []
    else:
        docs = [document_text(item) for item in older]
        vectors = []

        for start in range(0, len(docs), batch_size):
            vectors.extend(embed_fn(
                docs[start:start + batch_size],
                config,
                prefix="search_document",
            ))

        query = embed_fn(
            [query_text(secret)],
            config,
            prefix="search_query",
        )[0]

        query = normalize(query)
        ranked = []

        for item, vector in zip(older, vectors):
            similarity = cosine(
                query, normalize(vector)
            )
            ranked.append((
                similarity, item["turn"], item
            ))

        ranked.sort(key=lambda x: (-x[0], -x[1]))
        selected = sorted(
            (
                item for _, _, item
                in ranked[:evidence_turns]
            ),
            key=lambda item: item["turn"],
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
            "older_matching_candidates": len(older),
            "window_turns": window_turns,
            "evidence_turns": evidence_turns,
            "history_complete": covered == len(history),
            "retrieval_is_exhaustive": covered == len(history),
        },
    }
