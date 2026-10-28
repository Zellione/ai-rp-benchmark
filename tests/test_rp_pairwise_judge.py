"""Pairwise scheduling and evidence validation."""

from copy import deepcopy

import pytest

from rpbench.rp_pairwise_judge import (
    matchups,
    validate_dataset,
    validate_verdict,
)


def candidate(identifier):
    return {
        "id": identifier,
        "transcript": [
            {
                "turn": 1,
                "role": "player",
                "text": "Guten Abend.",
            },
            {
                "turn": 1,
                "role": "npc",
                "text": f"{identifier} schließt die Tür.",
            },
        ],
    }


def dataset():
    return {
        "schema_version": 1,
        "cases": [candidate(c) for c in "ABCD"],
    }


def verdict():
    return {
        "winner": "tie",
        "reason": "Comparable RP quality.",
        "x_strength": "Consistent voice",
        "x_weakness": "Limited initiative",
        "y_strength": "Good reactions",
        "y_weakness": "Formulaic imagery",
        "evidence_x": [
            {"turn": 1, "quote": "A schließt die Tür."}
        ],
        "evidence_y": [
            {"turn": 1, "quote": "B schließt die Tür."}
        ],
    }


def test_twelve_position_balanced_jobs():
    jobs = matchups(dataset())
    assert len(jobs) == 12
    assert len(set(jobs)) == 12

    for x, y in jobs:
        assert (y, x) in jobs


def test_seed_reproducible():
    assert matchups(dataset(), 987) == matchups(dataset(), 987)


def test_candidate_order_does_not_change_jobs():
    original = dataset()
    reversed_data = deepcopy(original)
    reversed_data["cases"].reverse()

    assert matchups(original, 99) == matchups(reversed_data, 99)


def test_valid_verdict():
    result = validate_verdict(
        verdict(), candidate("A"), candidate("B")
    )
    assert result["winner"] == "tie"


def test_fabricated_quote_rejected():
    proposal = verdict()
    proposal["evidence_x"][0]["quote"] = "Erfundene Szene."

    with pytest.raises(ValueError, match="Unsupported"):
        validate_verdict(
            proposal, candidate("A"), candidate("B")
        )


def test_player_quote_rejected():
    proposal = verdict()
    proposal["evidence_x"][0]["quote"] = "Guten Abend."

    with pytest.raises(ValueError, match="Unsupported"):
        validate_verdict(
            proposal, candidate("A"), candidate("B")
        )


def test_unknown_fields_rejected():
    proposal = verdict()
    proposal["confidence"] = 0.8

    with pytest.raises(ValueError, match="fields"):
        validate_verdict(
            proposal, candidate("A"), candidate("B")
        )


def test_duplicate_candidate_id_rejected():
    payload = dataset()
    payload["cases"].append(candidate("A"))

    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(payload)
