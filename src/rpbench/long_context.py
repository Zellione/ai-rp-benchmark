"""Deterministic, provenance-preserving long-context retrieval."""

from __future__ import annotations

import re


VERSION = "lexical-retrieval-v1"


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
