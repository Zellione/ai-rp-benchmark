from copy import deepcopy

import pytest

from rpbench.goal_criteria import (
    validate_goal_criteria,
    prerequisites_met,
    evaluation_prerequisites_met,
    required_goal_summary,
)
from rpbench.criteria_progress import advance_criteria_goals


def make_state():
    return {
        "schema_version": 1,
        "last_confirmed_event": 0,
        "goals": {
            "subgoal-01": {
                "description": "Optional investigation",
                "status": "pending",
            },
            "subgoal-02": {
                "description": "Recover the object",
                "status": "pending",
            },
            "subgoal-03": {
                "description": "Document the outcome",
                "status": "pending",
            },
        },
    }


def make_definition():
    goals = {}

    for index in range(1, 4):
        goal_id = f"subgoal-{index:02d}"

        goals[goal_id] = {
            "requires": (
                [f"subgoal-{index-1:02d}"]
                if index > 1 else []
            ),
            "required": index > 1,
            "evaluate_independently": index > 1,
            "criteria": [{
                "id": "evidence",
                "description": f"Evidence for {goal_id}",
                "evidence_contains": "evidence",
            }],
        }

    return {
        "schema_version": 1,
        "goals": goals,
    }


def test_optional_goal_does_not_block_evaluation():
    state = make_state()
    definition = make_definition()

    assert not prerequisites_met(
        definition, state, "subgoal-02"
    )

    assert evaluation_prerequisites_met(
        definition, state, "subgoal-02"
    )


def test_player_prerequisites_remain_unchanged():
    state = make_state()
    definition = make_definition()

    assert prerequisites_met(
        definition, state, "subgoal-01"
    )

    assert not prerequisites_met(
        definition, state, "subgoal-02"
    )

    state["goals"]["subgoal-01"]["status"] = "completed"

    assert prerequisites_met(
        definition, state, "subgoal-02"
    )


def test_required_goal_summary_allows_optional_unfinished():
    state = make_state()
    definition = make_definition()

    state["goals"]["subgoal-02"]["status"] = "completed"
    state["goals"]["subgoal-03"]["status"] = "completed"

    result = required_goal_summary(definition, state)

    assert result["complete"] is True
    assert result["missing_required_goals"] == []
    assert result["optional_goals"] == ["subgoal-01"]


def test_missing_required_goal_blocks_completion():
    state = make_state()
    definition = make_definition()

    state["goals"]["subgoal-02"]["status"] = "completed"

    result = required_goal_summary(definition, state)

    assert result["complete"] is False
    assert result["missing_required_goals"] == [
        "subgoal-03"
    ]


def test_legacy_configuration_remains_strict():
    state = make_state()
    definition = make_definition()

    for entry in definition["goals"].values():
        entry.pop("required")
        entry.pop("evaluate_independently")

    validate_goal_criteria(definition, state)

    assert not evaluation_prerequisites_met(
        definition, state, "subgoal-02"
    )

    state["goals"]["subgoal-02"]["status"] = "completed"
    state["goals"]["subgoal-03"]["status"] = "completed"

    result = required_goal_summary(definition, state)

    assert result["complete"] is False
    assert "subgoal-01" in result["missing_required_goals"]


def test_required_flag_requires_boolean():
    state = make_state()
    definition = make_definition()

    definition["goals"]["subgoal-01"]["required"] = "false"

    with pytest.raises(ValueError, match="required"):
        validate_goal_criteria(definition, state)


def test_independent_completion_replays_with_optional_open(
    monkeypatch,
):
    state = make_state()
    definition = make_definition()

    progress = {
        "schema_version": 1,
        "goals": {
            goal_id: {
                "proposed_satisfied": goal_id == "subgoal-02",
                "criteria": {
                    "evidence": {
                        "proposed_satisfied": goal_id == "subgoal-02",
                        "evidence": [],
                    },
                },
            }
            for goal_id in definition["goals"]
        },
    }

    monkeypatch.setattr(
        "rpbench.criteria_progress.replay_semantic_criteria",
        lambda *args: deepcopy(progress),
    )

    updated, _, transitions = advance_criteria_goals(
        state,
        definition,
        [],
        [],
    )

    assert updated["goals"]["subgoal-01"]["status"] == "pending"
    assert updated["goals"]["subgoal-02"]["status"] == "completed"
    assert updated["goals"]["subgoal-03"]["status"] == "pending"

    assert [
        item["goal_id"] for item in transitions
    ] == ["subgoal-02"]

    assert required_goal_summary(
        definition, updated
    )["complete"] is False
