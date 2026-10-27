"""Quest scheduling without LLM inference."""

from rpbench.quest_scheduler import plan_goal_evaluations


def make_state(*, request="pending", access="pending"):
    return {
        "goals": {
            "subgoal-07": {
                "description": "Request access",
                "status": request,
            },
            "subgoal-08": {
                "description": "Obtain access",
                "status": access,
            },
        },
    }


DEFINITION = {
    "goals": {
        "subgoal-07": {
            "criteria": [
                {"id": "authorization-request"},
            ],
        },
        "subgoal-08": {
            "criteria": [
                {"id": "authorized-access"},
                {"id": "document-examined"},
            ],
        },
    },
}


def make_progress(*, authorized=False, examined=False):
    return {
        "goals": {
            "subgoal-07": {
                "criteria": {
                    "authorization-request": {
                        "proposed_satisfied": False
                    },
                },
            },
            "subgoal-08": {
                "criteria": {
                    "authorized-access": {
                        "proposed_satisfied": authorized
                    },
                    "document-examined": {
                        "proposed_satisfied": examined
                    },
                },
            },
        },
    }


def test_request_is_active_and_access_is_candidate():
    plan = plan_goal_evaluations(
        make_state(), DEFINITION, make_progress()
    )

    assert plan["active_goal_id"] == "subgoal-07"
    assert plan["opportunity_candidates"] == [{
        "goal_id": "subgoal-08",
        "criterion_ids": [
            "authorized-access", "document-examined"
        ],
    }]


def test_completed_request_activates_access():
    plan = plan_goal_evaluations(
        make_state(request="completed"),
        DEFINITION,
        make_progress(),
    )

    assert plan["active_goal_id"] == "subgoal-08"
    assert plan["opportunity_candidates"] == []


def test_skipped_request_activates_access():
    plan = plan_goal_evaluations(
        make_state(request="skipped"),
        DEFINITION,
        make_progress(authorized=True),
    )

    assert plan["active_goal_id"] == "subgoal-08"
    assert plan["opportunity_candidates"] == []


def test_completed_goals_never_scheduled():
    plan = plan_goal_evaluations(
        make_state(request="completed", access="completed"),
        DEFINITION,
        make_progress(authorized=True, examined=True),
    )

    assert plan == {
        "active_goal_id": None,
        "opportunity_candidates": [],
    }


def test_only_unresolved_criteria_become_candidates():
    plan = plan_goal_evaluations(
        make_state(),
        DEFINITION,
        make_progress(authorized=True),
    )

    assert plan["opportunity_candidates"] == [{
        "goal_id": "subgoal-08",
        "criterion_ids": ["document-examined"],
    }]
