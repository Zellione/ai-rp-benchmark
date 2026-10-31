import pytest

from rpbench.goal_criteria import (
    validate_goal_criteria, prerequisites_met,
)
from rpbench.player_controller import select_player_goal
from rpbench.quest_scheduler import plan_goal_evaluations


def fixture():
    state = {
        "goals": {
            "subgoal-01": {
                "status": "pending",
                "description": "Identify the archive manager",
            },
            "subgoal-02": {
                "status": "pending",
                "description": "Obtain permission",
            },
            "subgoal-03": {
                "status": "pending",
                "description": "Inspect the register",
            },
        }
    }

    goals = {}
    for i in range(1, 4):
        key = f"subgoal-{i:02}"
        entry = {
            "criteria": [{
                "id": f"criterion-{i}",
                "description": f"Observable result {i}",
                "evidence_contains": f"result-{i}",
            }]
        }
        if i > 1:
            entry["requires"] = [f"subgoal-{i-1:02}"]
        goals[key] = entry

    definition = {"schema_version": 1, "goals": goals}
    progress = {
        "goals": {
            key: {
                "criteria": {
                    value["criteria"][0]["id"]: {
                        "proposed_satisfied": False
                    }
                }
            }
            for key, value in goals.items()
        }
    }
    return state, definition, progress


def test_only_first_goal_is_initially_eligible():
    state, definition, progress = fixture()
    validate_goal_criteria(definition, state)

    assert select_player_goal(state, definition)["id"] == "subgoal-01"
    assert not prerequisites_met(definition, state, "subgoal-02")
    assert not prerequisites_met(definition, state, "subgoal-03")

    plan = plan_goal_evaluations(state, definition, progress)
    assert plan["active_goal_id"] == "subgoal-01"
    assert plan["opportunity_candidates"] == []


def test_next_goal_unlocks_after_verified_completion():
    state, definition, _ = fixture()
    state["goals"]["subgoal-01"]["status"] = "completed"

    assert select_player_goal(state, definition)["id"] == "subgoal-02"
    assert prerequisites_met(definition, state, "subgoal-02")
    assert not prerequisites_met(definition, state, "subgoal-03")


def test_cycle_is_rejected():
    state, definition, _ = fixture()
    definition["goals"]["subgoal-01"]["requires"] = ["subgoal-03"]

    with pytest.raises(ValueError, match="Cyclic"):
        validate_goal_criteria(definition, state)


def test_unknown_dependency_is_rejected():
    state, definition, _ = fixture()
    definition["goals"]["subgoal-02"]["requires"] = ["missing"]

    with pytest.raises(ValueError, match="prerequisites"):
        validate_goal_criteria(definition, state)


def test_legacy_goals_remain_eligible():
    state, definition, progress = fixture()
    for entry in definition["goals"].values():
        entry.pop("requires", None)

    validate_goal_criteria(definition, state)
    plan = plan_goal_evaluations(state, definition, progress)

    assert len(plan["opportunity_candidates"]) == 2
