import pytest

from rpbench.goal_criteria import (
    prerequisites_met,
    validate_goal_criteria,
)
from rpbench.player_controller import select_player_goal


def make_state():
    return {
        "goals": {
            "subgoal-01": {
                "description": "Identify the archivist",
                "status": "completed",
            },
            "subgoal-02": {
                "description": "Understand the restricted records",
                "status": "pending",
            },
            "subgoal-03": {
                "description": "Obtain permission",
                "status": "pending",
            },
        },
    }


def make_definition():
    return {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [{
                    "id": "identity",
                    "description": "Identify the archivist",
                    "evidence_contains": "archivist",
                }],
            },
            "subgoal-03": {
                "requires": ["subgoal-01"],
                "criteria": [{
                    "id": "permission",
                    "description": "Obtain explicit permission",
                    "evidence_contains": "permission",
                }],
            },
        },
    }


def test_legacy_goal_without_criteria_remains_eligible():
    state = make_state()
    definition = make_definition()

    validate_goal_criteria(definition, state)

    assert prerequisites_met(
        definition, state, "subgoal-02"
    )

    selected = select_player_goal(state, definition)
    assert selected["id"] == "subgoal-02"


def test_defined_prerequisites_still_apply():
    state = make_state()
    definition = make_definition()

    state["goals"]["subgoal-01"]["status"] = "pending"

    assert not prerequisites_met(
        definition, state, "subgoal-03"
    )

    state["goals"]["subgoal-01"]["status"] = "completed"

    assert prerequisites_met(
        definition, state, "subgoal-03"
    )


def test_unknown_player_goal_is_rejected():
    with pytest.raises(ValueError, match="Unknown player goal"):
        prerequisites_met(
            make_definition(),
            make_state(),
            "nonexistent",
        )
