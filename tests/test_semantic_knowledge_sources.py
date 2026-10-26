"""Semantic knowledge with source-turn provenance."""

from rpbench.evaluation_journal import (
    make_evaluation_event,
)
from rpbench.player_evaluator import validate_evaluation


NPC = (
    "For special authorizations, please speak with "
    "our curator, Mr. Alden Voss."
)

STATE = {
    "goals": {
        "subgoal-01": {
            "description": "Identify archive staff",
            "status": "pending",
        },
    },
}

PROPOSAL = {
    "knowledge_updates": [{
        "fact": (
            "Mara directs researchers seeking special "
            "authorization to curator Alden Voss."
        ),
        "source_turns": [2],
    }],
    "goal_updates": [],
}


def test_semantic_paraphrase_accepted():
    result = validate_evaluation(
        PROPOSAL, NPC, STATE, turn=2
    )
    assert result["knowledge_updates"][0]["status"] == "reported"
    assert result["knowledge_updates"][0]["source_turns"] == [2]


def test_wrong_source_turn_rejected():
    from copy import deepcopy
    import pytest

    changed = deepcopy(PROPOSAL)
    changed["knowledge_updates"][0]["source_turns"] = [9]

    with pytest.raises(ValueError, match="source turn"):
        validate_evaluation(
            changed, NPC, STATE, turn=2
        )


def test_semantic_event_preserves_source():
    event = make_evaluation_event(
        2, NPC, STATE, PROPOSAL, "semantic-evaluator"
    )
    assert event["result"]["knowledge_updates"][0][
        "source_turns"
    ] == [2]
