"""Approval and deterministic reconstruction tests."""

from copy import deepcopy

import pytest

from rpbench.goal_progress import (
    replay_goal_progress,
    validate_goal_rules,
)


def initial_state():
    return {
        "last_confirmed_event": 0,
        "goals": {
            "subgoal-01": {
                "description": "Identify archive staff",
                "status": "pending",
            },
            "subgoal-02": {
                "description": "Understand restrictions",
                "status": "pending",
            },
        },
    }


def evaluations():
    return {
        1: {
            "turn": 1,
            "result": {
                "knowledge_updates": [],
                "goal_updates": [{
                    "goal_id": "subgoal-01",
                    "status": "completed",
                    "evidence": (
                        "Mara Vey is the senior archivist."
                    ),
                }],
            },
        },
    }


RULES = [{
    "goal_id": "subgoal-01",
    "evidence_contains": "senior archivist",
}]


def test_approved_goal_advances():
    state, approved = replay_goal_progress(
        initial_state(), evaluations(), RULES
    )

    assert state["goals"]["subgoal-01"]["status"] == "completed"
    assert state["goals"]["subgoal-02"]["status"] == "pending"
    assert state["last_confirmed_event"] == 1
    assert approved[0]["turn"] == 1


def test_no_rule_means_no_approval():
    state, approved = replay_goal_progress(
        initial_state(), evaluations(), []
    )

    assert state == initial_state()
    assert approved == []


def test_evidence_must_match():
    state, approved = replay_goal_progress(
        initial_state(),
        evaluations(),
        [{
            "goal_id": "subgoal-01",
            "evidence_contains": "authorization granted",
        }],
    )

    assert state["last_confirmed_event"] == 0
    assert approved == []


def test_replay_is_deterministic_and_nonmutating():
    initial = initial_state()
    events = evaluations()
    before = deepcopy((initial, events))

    first = replay_goal_progress(initial, events, RULES)
    second = replay_goal_progress(initial, events, RULES)

    assert first == second
    assert (initial, events) == before


def test_duplicate_proposal_does_not_advance_twice():
    events = evaluations()
    events[2] = deepcopy(events[1])
    events[2]["turn"] = 2

    state, approved = replay_goal_progress(
        initial_state(), events, RULES
    )

    assert state["last_confirmed_event"] == 1
    assert len(approved) == 1


def test_unknown_goal_rule_rejected():
    with pytest.raises(ValueError):
        validate_goal_rules(
            [{
                "goal_id": "missing",
                "evidence_contains": "something",
            }],
            initial_state(),
        )
