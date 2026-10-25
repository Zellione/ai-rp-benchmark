"""Tests for deterministic controlled-player goal selection."""

import copy

import pytest

from rpbench.player_controller import (
    build_controlled_player_instruction,
    select_player_goal,
)


def state():
    return {
        "goals": {
            "subgoal-01": {
                "description": "Gain access",
                "status": "pending",
            },
            "subgoal-02": {
                "description": "Read records",
                "status": "pending",
            },
            "subgoal-03": {
                "description": "Find the truth",
                "status": "pending",
            },
        },
    }


def test_first_pending_goal():
    assert select_player_goal(state())["id"] == "subgoal-01"


def test_active_goal_takes_priority():
    value = state()
    value["goals"]["subgoal-03"]["status"] = "active"

    assert select_player_goal(value)["id"] == "subgoal-03"


def test_completed_and_failed_goals_are_skipped():
    value = state()
    value["goals"]["subgoal-01"]["status"] = "completed"
    value["goals"]["subgoal-02"]["status"] = "failed"

    assert select_player_goal(value)["id"] == "subgoal-03"


def test_no_open_goals():
    value = state()
    for goal in value["goals"].values():
        goal["status"] = "completed"

    assert select_player_goal(value) is None


def test_instruction_preserves_goal_state():
    value = state()
    before = copy.deepcopy(value)

    instruction = build_controlled_player_instruction(value)

    assert "Gain access" in instruction
    assert "actual NPC response" in instruction
    assert value == before


@pytest.mark.parametrize("invalid", [
    None,
    {},
    {"goals": []},
    {"goals": {"x": {"status": "unknown",
                     "description": "Find clues"}}},
])
def test_invalid_state_rejected(invalid):
    with pytest.raises(ValueError):
        select_player_goal(invalid)
