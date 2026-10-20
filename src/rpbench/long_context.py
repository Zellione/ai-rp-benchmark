"""Deterministic, provenance-preserving long-context retrieval."""

from __future__ import annotations

import re


VERSION = "lexical-retrieval-v2"


def tokens(text):
    if not isinstance(text, str):
        raise ValueError("Expected text")
    return set(re.findall(r"\w+", text.casefold()))


def validate_settings(window_turns, evidence_turns):
    if type(window_turns) is not int or window_turns < 1:
        raise ValueError("window_turns must be positive")
    if type(evidence_turns) is not int or evidence_turns < 0:
        raise ValueError("evidence_turns must be nonnegative")


def relevance_terms(secret):
    """Build a reproducible query from the secret specification."""
    terms = tokens(secret["fact"])
    for marker in secret.get("exact_markers", []):
        terms.update(tokens(marker))
    for marker in secret.get("context_markers", []):
        terms.update(tokens(marker))

    # Remove common function words, not story-specific vocabulary.
    stop = {
        "a", "an", "the", "is", "in", "on", "at",
        "to", "of", "it", "and", "or", "from",
        "for", "this", "that", "with", "beneath",
        "under", "where", "who", "was", "are",
        "der", "die", "das", "ein", "eine",
        "und", "ist", "im", "am", "zu", "von",
    }
    return terms - stop


def select_context(
    history,
    secret,
    *,
    window_turns=12,
    evidence_turns=8,
):
    """Select complete prior turns, never future or partial turns.

    The recent window is retained without semantic filtering.
    Older turns are ranked by a deterministic lexical score.
    A zero-scoring older turn is not presented as relevant.
    """
    validate_settings(window_turns, evidence_turns)

    turns = [item["turn"] for item in history]
    if any(type(t) is not int or t < 1 for t in turns):
        raise ValueError("Invalid history turn")
    if turns != sorted(set(turns)):
        raise ValueError("History must be ordered and unique")
    if any(
        not isinstance(item.get("player"), str)
        or not isinstance(item.get("npc"), str)
        for item in history
    ):
        raise ValueError("Incomplete history entry")

    recent = history[-window_turns:]
    older = history[:-window_turns]
    query = relevance_terms(secret)

    candidates = []
    for item in older:
        player_terms = tokens(item["player"])
        npc_terms = tokens(item["npc"])
        # NPC statements receive a slight preference,
        # but player-originated knowledge is also searchable.
        score = (
            len(query & player_terms)
            + 2 * len(query & npc_terms)
        )
        if score > 0:
            candidates.append((score, item["turn"], item))

    ranked = sorted(
        candidates,
        key=lambda x: (-x[0], -x[1]),
    )[:evidence_turns]

    selected = sorted(
        (entry for _, _, entry in ranked),
        key=lambda item: item["turn"],
    )

    total = len(history)
    covered = len(recent) + len(selected)

    return {
        "retrieval_version": VERSION,
        "recent_context": list(recent),
        "retrieved_older_evidence": selected,
        "coverage": {
            "total_prior_turns": total,
            "included_prior_turns": covered,
            "omitted_prior_turns": total - covered,
            "older_matching_candidates": len(candidates),
            "window_turns": window_turns,
            "evidence_turns": evidence_turns,
            "history_complete": covered == total,
            "retrieval_is_exhaustive": covered == total,
        },
    }


def fit_payload_budget(job, make_payload, config, max_bytes):
    """Fit selected history into an exact serialized request byte budget.

    Never truncate the target turn, secret, or system instructions.
    Return a derived job; do not mutate the caller's original job.
    """
    import copy
    import json

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    selected = copy.deepcopy(job)
    context = selected["context_selection"]
    recent = context["recent_context"]
    older = context["retrieved_older_evidence"]

    original_turns = sorted(
        item["turn"] for item in recent + older
    )
    removed = []

    def payload_size():
        payload = make_payload(selected, config)
        return len(
            json.dumps(
                payload, ensure_ascii=False
            ).encode("utf-8")
        )

    initial_bytes = payload_size()

    while payload_size() > max_bytes:
        if older:
            query = relevance_terms(selected["secret"])

            def evidence_rank(item):
                score = (
                    len(query & tokens(item["player"]))
                    + 2 * len(query & tokens(item["npc"]))
                )
                # Lower score is discarded first.
                # For equal scores, older evidence goes first.
                return (score, item["turn"])

            weakest = min(
                range(len(older)),
                key=lambda index: evidence_rank(older[index]),
            )
            removed.append(older.pop(weakest)["turn"])
        elif recent:
            removed.append(recent.pop(0)["turn"])
        else:
            raise ValueError(
                "Target turn, secret, and instructions exceed "
                "the request byte budget"
            )

    covered = len(recent) + len(older)
    coverage = context["coverage"]
    coverage["included_prior_turns"] = covered
    coverage["omitted_prior_turns"] = (
        coverage["total_prior_turns"] - covered
    )
    coverage["history_complete"] = (
        coverage["omitted_prior_turns"] == 0
    )
    coverage["retrieval_is_exhaustive"] = (
        coverage["omitted_prior_turns"] == 0
    )

    final_bytes = payload_size()
    context["budget_audit"] = {
        "unit": "serialized_utf8_bytes",
        "max_bytes": max_bytes,
        "initial_bytes": initial_bytes,
        "final_bytes": final_bytes,
        "removed_turns": sorted(removed),
        "retained_turns": sorted(
            item["turn"] for item in recent + older
        ),
        "initial_selected_turns": original_turns,
        "token_count_verified": False,
    }

    return selected


def count_ninfer_prompt_tokens(payload, config):
    """Count the matching text-only prompt via ninfer Responses API."""
    import json
    import os
    import urllib.request

    messages = payload["messages"]
    if (
        len(messages) != 2
        or messages[0]["role"] != "system"
        or messages[1]["role"] != "user"
        or any(not isinstance(m["content"], str) for m in messages)
    ):
        raise ValueError(
            "Token counter requires system+user text messages"
        )

    body = {
        "model": payload["model"],
        "instructions": messages[0]["content"],
        "input": messages[1]["content"],
    }

    backend = config["backend"]
    url = (
        backend["base_url"].rstrip("/")
        + "/v1/responses/input_tokens"
    )

    headers = {"Content-Type": "application/json"}
    key = os.getenv(
        backend.get("api_key_env", "RPBENCH_JUDGE_API_KEY")
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(
            body, ensure_ascii=False
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=backend.get("timeout", 600)
    ) as response:
        result = json.load(response)

    count = result.get("input_tokens")
    if type(count) is not int or count < 1:
        raise ValueError("Invalid ninfer input token count")

    return count


def fit_token_budget(
    job,
    make_payload,
    config,
    max_prompt_tokens,
    *,
    counter=count_ninfer_prompt_tokens,
):
    """Constrain a derived job using server-side prompt token counts."""
    from copy import deepcopy

    if (
        type(max_prompt_tokens) is not int
        or max_prompt_tokens < 1
    ):
        raise ValueError("Invalid max_prompt_tokens")

    selected = deepcopy(job)
    context = selected["context_selection"]
    recent = context["recent_context"]
    older = context["retrieved_older_evidence"]

    removed = []

    def count():
        return counter(make_payload(selected, config), config)

    initial = count()
    current = initial

    while current > max_prompt_tokens:
        if older:
            query = relevance_terms(selected["secret"])

            def rank(item):
                return (
                    len(query & tokens(item["player"]))
                    + 2 * len(query & tokens(item["npc"])),
                    item["turn"],
                )

            index = min(
                range(len(older)),
                key=lambda i: rank(older[i]),
            )
            removed.append(older.pop(index)["turn"])
        elif recent:
            removed.append(recent.pop(0)["turn"])
        else:
            raise ValueError(
                "Target, secret and instructions exceed "
                "the prompt token budget"
            )

        current = count()

    coverage = context["coverage"]
    covered = len(recent) + len(older)
    coverage["included_prior_turns"] = covered
    coverage["omitted_prior_turns"] = (
        coverage["total_prior_turns"] - covered
    )
    coverage["history_complete"] = (
        coverage["omitted_prior_turns"] == 0
    )
    coverage["retrieval_is_exhaustive"] = (
        coverage["omitted_prior_turns"] == 0
    )

    context["token_budget_audit"] = {
        "counter": "ninfer-responses-input-tokens",
        "max_prompt_tokens": max_prompt_tokens,
        "initial_prompt_tokens": initial,
        "final_prompt_tokens": current,
        "removed_turns": sorted(removed),
        "retained_turns": sorted(
            t["turn"] for t in recent + older
        ),
        "server_counted": True,
    }

    return selected
