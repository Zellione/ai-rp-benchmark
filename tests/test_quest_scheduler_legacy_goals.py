"""Player goals without criteria must never become evaluator targets."""

from rpbench.player_controller import select_player_goal
from rpbench.quest_scheduler import plan_goal_evaluations


def make_case():
    state = {
        "goals": {
            "subgoal-01": {
                "description": "Find Mara",
                "status": "completed",
            },
            "subgoal-02": {
                "description": "Read the register",
                "status": "pending",
            },
        },
    }

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [{
                    "id": "found",
                    "description": "Find Mara",
                    "evidence_contains": "Mara",
                }],
            },
        },
    }

    progress = {
        "goals": {
            "subgoal-01": {
                "criteria": {
                    "found": {
                        "proposed_satisfied": True,
                    }
                },
            },
        },
    }

    return state, definition, progress


def test_unconfigured_goal_remains_player_objective():
    state, definition, _ = make_case()

    selected = select_player_goal(state, definition)

    assert selected is not None
    assert selected["id"] == "subgoal-02"


def test_unconfigured_goal_not_sent_for_evaluation():
    state, definition, progress = make_case()

    plan = plan_goal_evaluations(
        state, definition, progress
    )

    assert plan["active_goal_id"] is None
    assert plan["opportunity_candidates"] == []


def test_configured_goal_still_scheduled():
    state, definition, progress = make_case()

    state["goals"]["subgoal-01"]["status"] = "pending"
    progress["goals"]["subgoal-01"]["criteria"][
        "found"
    ]["proposed_satisfied"] = False

    plan = plan_goal_evaluations(
        state, definition, progress
    )

    assert plan["active_goal_id"] == "subgoal-01"
