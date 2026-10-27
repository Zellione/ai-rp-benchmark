"""Strict, conservative normalization of knowledge proposals."""

import pytest

from rpbench.knowledge_normalization import (
    normalize_knowledge_proposal,
)
from rpbench.player_evaluator import validate_evaluation


VISIBLE = (
    "Finn tells Elena that Voss is a custodian "
    "and Mara manages the archive."
)

STATE = {"goals": {}}


def check(proposal):
    return validate_evaluation(
        proposal, VISIBLE, STATE, turn=4
    )


def test_singular_source_turn_array():
    proposal = {
        "knowledge_updates": [{
            "fact": "Finn describes Voss as a custodian.",
            "source_turn": [4],
        }],
        "goal_updates": [],
    }

    result = normalize_knowledge_proposal(
        proposal, VISIBLE, 4
    )

    assert result["knowledge_updates"][0] == {
        "fact": "Finn describes Voss as a custodian.",
        "source_turns": [4],
    }
    assert check(result)["knowledge_updates"][0][
        "status"
    ] == "reported"


def test_valid_extra_evidence_is_removed():
    proposal = {
        "knowledge_updates": [{
            "fact": "Finn describes Voss as a custodian.",
            "source_turns": [4],
            "evidence": "Voss is a custodian",
        }],
        "goal_updates": [],
    }

    result = normalize_knowledge_proposal(
        proposal, VISIBLE, 4
    )

    assert set(result["knowledge_updates"][0]) == {
        "fact", "source_turns"
    }
    check(result)


@pytest.mark.parametrize("entry", [
    {
        "fact": "Unsupported fact",
        "source_turns": [3],
        "evidence": "Voss is a custodian",
    },
    {
        "fact": "Unsupported fact",
        "source_turns": [4],
        "evidence": "Invented quotation",
    },
    {
        "fact": "Unsupported fact",
        "source_turns": [4],
        "confidence": 0.9,
    },
    {
        "fact": "Unsupported fact",
        "source_turn": 4,
    },
])
def test_invalid_deviations_remain_invalid(entry):
    proposal = {
        "knowledge_updates": [entry],
        "goal_updates": [],
    }

    result = normalize_knowledge_proposal(
        proposal, VISIBLE, 4
    )

    with pytest.raises(ValueError):
        check(result)
