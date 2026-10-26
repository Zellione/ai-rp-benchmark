"""Integration of semantic criteria with player goal progression."""

from copy import deepcopy

from rpbench.criteria_progress import advance_criteria_goals
from rpbench.player_controller import select_player_goal
from rpbench.semantic_criteria import make_semantic_event


def test_player_advances_after_multiturn_criteria():
    state = {
        "last_confirmed_event": 0,
        "goals": {
            "subgoal-01": {
                "description": "Identify archive staff",
                "status": "pending",
            },
            "subgoal-02": {
                "description": "Investigate restricted records",
                "status": "pending",
            },
        },
    }

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [
                    {
                        "id": "mara",
                        "description": "Identify Mara's role",
                        "evidence_contains": "Senior Archivist",
                    },
                    {
                        "id": "finn",
                        "description": "Identify Finn's role",
                        "evidence_contains": "digitization",
                    },
                ],
            },
        },
    }

    observations = [
        {
            "turn": 1,
            "role": "npc",
            "text": "I am Mara, Senior Archivist.",
        },
        {
            "turn": 2,
            "role": "npc",
            "text": "Finn handles digitization.",
        },
    ]

    def event(turn, criterion, quote):
        proposal = {
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-01",
                "criterion_id": criterion,
                "status": "satisfied",
                "evidence": [{
                    "turn": turn,
                    "quote": quote,
                }],
            }],
        }
        return make_semantic_event(
            turn, proposal, definition, state,
            observations, "test-adjudicator",
        )

    first = event(1, "mara", "Senior Archivist")
    second = event(2, "finn", "Finn handles digitization.")

    assert select_player_goal(state)["id"] == "subgoal-01"

    partial, progress, approvals = advance_criteria_goals(
        state, definition, observations, [first]
    )
    assert partial["goals"]["subgoal-01"]["status"] == "pending"
    assert approvals == []

    original = deepcopy(state)

    advanced, progress, approvals = advance_criteria_goals(
        state, definition, observations, [first, second]
    )

    assert state == original
    assert advanced["goals"]["subgoal-01"]["status"] == "completed"
    assert advanced["goals"]["subgoal-02"]["status"] == "pending"
    assert select_player_goal(advanced)["id"] == "subgoal-02"
    assert advanced["last_confirmed_event"] == 1
    assert len(approvals) == 1
    assert approvals[0]["authority"] == "semantic_model_adjudicator"

    replayed, _, _ = advance_criteria_goals(
        state, definition, observations, [first, second]
    )
    assert replayed == advanced
