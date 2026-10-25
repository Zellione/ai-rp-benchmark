"""Untrusted semantic evaluator output contract."""

import pytest

from rpbench.player_evaluator import validate_evaluation


NPC = "Director Vale manages the restricted collection."

STATE = {
    "goals": {
        "subgoal-01": {
            "description": "Identify the responsible official",
            "status": "active",
        },
    },
}


def result():
    return {
        "knowledge_updates": [{
            "fact": "Director Vale manages restricted records.",
            "evidence": NPC,
        }],
        "goal_updates": [{
            "goal_id": "subgoal-01",
            "status": "completed",
            "evidence": NPC,
        }],
    }


def test_valid_proposal():
    clean = validate_evaluation(result(), NPC, STATE)
    assert clean["knowledge_updates"][0]["status"] == "reported"
    assert clean["goal_updates"][0]["status"] == "completed"


def test_no_updates():
    assert validate_evaluation(
        {"knowledge_updates": [], "goal_updates": []},
        NPC,
        STATE,
    ) == {"knowledge_updates": [], "goal_updates": []}


def test_hallucinated_evidence_rejected():
    value = result()
    value["knowledge_updates"][0]["evidence"] = "Not said"

    with pytest.raises(ValueError, match="evidence"):
        validate_evaluation(value, NPC, STATE)


def test_unknown_goal_rejected():
    value = result()
    value["goal_updates"][0]["goal_id"] = "invented"

    with pytest.raises(ValueError, match="goal"):
        validate_evaluation(value, NPC, STATE)


def test_untrusted_extra_fields_rejected():
    value = result()
    value["knowledge_updates"][0]["verified"] = True

    with pytest.raises(ValueError):
        validate_evaluation(value, NPC, STATE)


def test_terminal_goal_rejected():
    state = {
        "goals": {
            "subgoal-01": {
                "description": "Identify official",
                "status": "completed",
            },
        },
    }

    with pytest.raises(ValueError, match="terminal"):
        validate_evaluation(result(), NPC, state)
