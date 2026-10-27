"""Optional quest steps are skipped by real later success."""

from rpbench.criteria_progress import advance_criteria_goals
from rpbench.goal_criteria import validate_goal_criteria
from rpbench.player_controller import select_player_goal
from rpbench.semantic_criteria import make_semantic_event
from rpbench.state import initial_state


GROUND_TRUTH = {
    "schema_version": 1,
    "player": {
        "subgoals": {
            "subgoal-07": {
                "description": "Request access",
                "initial_status": "pending",
            },
            "subgoal-08": {
                "description": "Obtain access and inspect documents",
                "initial_status": "pending",
            },
        },
    },
    "characters": {},
}

DEFINITION = {
    "schema_version": 1,
    "goals": {
        "subgoal-07": {
            "criteria": [{
                "id": "authorization-request",
                "description": "Submit a restricted access request",
                "evidence_contains": "request",
            }],
            "superseded_by": {
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
            },
        },
        "subgoal-08": {
            "criteria": [{
                "id": "authorized-access",
                "description": "Official authorization granted",
                "evidence_contains": "authorization",
            }, {
                "id": "document-examined",
                "description": "Restricted document actually examined",
                "evidence_contains": "document",
            }],
        },
    },
}

OBSERVATIONS = [
    {"role": "player", "turn": 1, "text": "I request access."},
    {"role": "npc", "turn": 1, "text": "Voss responds."},
]


def evaluate_criterion(criterion_id):
    state = initial_state(GROUND_TRUTH)
    proposal = {
        "schema_version": 1,
        "criterion_updates": [{
            "goal_id": "subgoal-08",
            "criterion_id": criterion_id,
            "status": "satisfied",
            "source_turns": [1],
            "reason": "Confirmed by the adjudicator.",
        }],
    }

    event = make_semantic_event(
        1, proposal, DEFINITION, state, OBSERVATIONS,
        "quest-test", goal_id="subgoal-08",
    )

    return advance_criteria_goals(
        state, DEFINITION, OBSERVATIONS, [event]
    )


def test_authorization_supersedes_request():
    state, progress, transitions = evaluate_criterion(
        "authorized-access"
    )

    assert state["goals"]["subgoal-07"]["status"] == "skipped"
    assert state["goals"]["subgoal-08"]["status"] == "pending"
    assert progress["goals"]["subgoal-08"]["criteria"][
        "authorized-access"
    ]["proposed_satisfied"]

    assert transitions[0]["status"] == "skipped"
    assert transitions[0]["authority"] == "quest_dependency_rule"
    assert select_player_goal(state)["id"] == "subgoal-08"


def test_unauthorized_examination_does_not_skip_request():
    state, progress, transitions = evaluate_criterion(
        "document-examined"
    )

    assert state["goals"]["subgoal-07"]["status"] == "pending"
    assert state["goals"]["subgoal-08"]["status"] == "pending"
    assert select_player_goal(state)["id"] == "subgoal-07"
    assert transitions == []


def test_unknown_superseding_criterion_rejected():
    state = initial_state(GROUND_TRUTH)
    invalid = {
        **DEFINITION,
        "goals": {
            **DEFINITION["goals"],
            "subgoal-07": {
                **DEFINITION["goals"]["subgoal-07"],
                "superseded_by": {
                    "goal_id": "subgoal-08",
                    "criterion_id": "nonexistent",
                },
            },
        },
    }

    import pytest
    with pytest.raises(
        ValueError, match="Unknown superseding criterion"
    ):
        validate_goal_criteria(invalid, state)
