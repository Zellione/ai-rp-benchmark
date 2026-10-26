"""Semantic source references without exact quote matching."""

import pytest

from rpbench.semantic_criteria import (
    make_semantic_event,
    replay_semantic_criteria,
)


STATE = {
    "last_confirmed_event": 0,
    "goals": {
        "subgoal-01": {
            "description": "Learn staff roles",
            "status": "pending",
        },
    },
}

DEFINITION = {
    "schema_version": 1,
    "goals": {
        "subgoal-01": {
            "criteria": [{
                "id": "voss-role",
                "description": "Identify access authority",
                "evidence_contains": "Voss",
            }],
        },
    },
}

OBSERVATIONS = [
    {
        "turn": 1,
        "role": "player",
        "text": "Who can authorize access?",
    },
    {
        "turn": 1,
        "role": "npc",
        "text": "Please speak to our curator, Mr. Alden Voss.",
    },
]


def proposal(status, turns=(1,)):
    return {
        "schema_version": 1,
        "criterion_updates": [{
            "goal_id": "subgoal-01",
            "criterion_id": "voss-role",
            "status": status,
            "source_turns": list(turns),
            "reason": (
                "The NPC identifies Voss as the relevant "
                "person for Elena to contact."
            ),
        }],
    }


def test_semantic_paraphrase_accepted():
    event = make_semantic_event(
        1, proposal("satisfied"),
        DEFINITION, STATE, OBSERVATIONS, "semantic-test",
        goal_id="subgoal-01",
    )

    progress = replay_semantic_criteria(
        DEFINITION, STATE, OBSERVATIONS, [event]
    )

    assert progress["goals"]["subgoal-01"][
        "proposed_satisfied"
    ]


def test_partial_does_not_complete_goal():
    event = make_semantic_event(
        1, proposal("partial"),
        DEFINITION, STATE, OBSERVATIONS, "semantic-test",
        goal_id="subgoal-01",
    )

    progress = replay_semantic_criteria(
        DEFINITION, STATE, OBSERVATIONS, [event]
    )

    criterion = progress["goals"]["subgoal-01"][
        "criteria"
    ]["voss-role"]

    assert not criterion["proposed_satisfied"]
    assert criterion["evidence"][0]["status"] == "partial"


def test_future_source_turn_rejected():
    with pytest.raises(ValueError, match="source turns"):
        make_semantic_event(
            1, proposal("satisfied", turns=(2,)),
            DEFINITION, STATE, OBSERVATIONS, "semantic-test",
            goal_id="subgoal-01",
        )
