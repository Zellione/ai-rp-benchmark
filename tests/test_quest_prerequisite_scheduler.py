from rpbench.goal_criteria import validate_goal_criteria
from rpbench.player_controller import select_player_goal
from rpbench.quest_scheduler import plan_goal_evaluations


def make_case():
    state = {
        "goals": {
            f"subgoal-{i:02}": {
                "status": "pending",
                "description": f"Observable objective {i}",
            }
            for i in range(1, 13)
        }
    }

    definitions = {}
    progress = {"goals": {}}

    for i in range(1, 13):
        goal_id = f"subgoal-{i:02}"
        criterion_id = f"evidence-{i:02}"

        definition = {
            "criteria": [{
                "id": criterion_id,
                "description": f"Observable event {i}",
                "evidence_contains": f"event-{i}",
            }],
        }

        if i > 1:
            definition["requires"] = [
                f"subgoal-{i-1:02}"
            ]

        definitions[goal_id] = definition
        progress["goals"][goal_id] = {
            "criteria": {
                criterion_id: {
                    "proposed_satisfied": False,
                }
            }
        }

    criteria = {
        "schema_version": 1,
        "goals": definitions,
    }

    validate_goal_criteria(criteria, state)
    return state, criteria, progress


def test_initially_only_first_goal_is_eligible():
    state, criteria, progress = make_case()

    selected = select_player_goal(state, criteria)
    plan = plan_goal_evaluations(
        state, criteria, progress
    )

    assert selected["id"] == "subgoal-01"
    assert plan["active_goal_id"] == "subgoal-01"
    assert plan["opportunity_candidates"] == []


def test_goals_unlock_sequentially():
    state, criteria, progress = make_case()

    for index in range(1, 12):
        goal_id = f"subgoal-{index:02}"
        next_id = f"subgoal-{index+1:02}"

        assert select_player_goal(
            state, criteria
        )["id"] == goal_id

        state["goals"][goal_id]["status"] = "completed"

        plan = plan_goal_evaluations(
            state, criteria, progress
        )

        assert plan["active_goal_id"] == next_id
        assert plan["opportunity_candidates"] == []

    state["goals"]["subgoal-12"]["status"] = "completed"
    assert select_player_goal(state, criteria) is None


def test_locked_goal_does_not_become_opportunity():
    state, criteria, progress = make_case()

    # Even if a later criterion has evidence, its prerequisites
    # are not satisfied and it must not be scheduled.
    progress["goals"]["subgoal-09"]["criteria"][
        "evidence-09"
    ]["proposed_satisfied"] = True

    plan = plan_goal_evaluations(
        state, criteria, progress
    )

    assert plan["active_goal_id"] == "subgoal-01"
    assert plan["opportunity_candidates"] == []
