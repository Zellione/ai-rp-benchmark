"""Versioned goal-criteria evaluation tests."""

from copy import deepcopy

import pytest

from rpbench.goal_criteria import (
    replay_goal_criteria,
    validate_goal_criteria,
)


def state():
    return {
        "last_confirmed_event": 0,
        "goals": {
            "subgoal-01": {
                "description": "Identify the archive staff",
                "status": "pending",
            },
        },
    }


def criteria():
    return {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [
                    {
                        "id": "identify-mara",
                        "description": "Learn Mara's role",
                        "evidence_contains": "Senior Archivist",
                    },
                    {
                        "id": "identify-finn",
                        "description": "Learn Finn's role",
                        "evidence_contains": "digitization",
                    },
                ],
            },
        },
    }


def event(turn, quote):
    return {
        "turn": turn,
        "result": {
            "knowledge_updates": [{
                "fact": "NPC statement",
                "evidence": quote,
                "status": "reported",
            }],
            "goal_updates": [],
        },
    }


def test_partial_progress():
    result = replay_goal_criteria(
        criteria(),
        state(),
        {1: event(1, "I am Mara, Senior Archivist.")},
    )

    goal = result["goals"]["subgoal-01"]
    assert not goal["satisfied"]
    assert goal["criteria"]["identify-mara"]["satisfied"]
    assert not goal["criteria"]["identify-finn"]["satisfied"]


def test_evidence_across_multiple_turns():
    result = replay_goal_criteria(
        criteria(),
        state(),
        {
            1: event(1, "I am Mara, Senior Archivist."),
            4: event(4, "I'm the digitization guy!"),
        },
    )

    goal = result["goals"]["subgoal-01"]
    assert goal["satisfied"]
    assert (
        goal["criteria"]["identify-finn"]["evidence"][0]["turn"]
        == 4
    )


def test_progress_does_not_complete_goal():
    original = state()

    replay_goal_criteria(
        criteria(),
        original,
        {
            1: event(1, "Senior Archivist"),
            2: event(2, "digitization"),
        },
    )

    assert original["goals"]["subgoal-01"]["status"] == "pending"
    assert original["last_confirmed_event"] == 0


def test_deterministic_and_nonmutating():
    definition = criteria()
    current = state()
    evaluations = {1: event(1, "Senior Archivist")}

    before = deepcopy((definition, current, evaluations))

    assert replay_goal_criteria(
        definition, current, evaluations
    ) == replay_goal_criteria(
        definition, current, evaluations
    )

    assert (definition, current, evaluations) == before


def test_invalid_schema_rejected():
    definition = criteria()
    definition["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        validate_goal_criteria(definition, state())


def test_duplicate_criterion_rejected():
    definition = criteria()
    first = deepcopy(
        definition["goals"]["subgoal-01"]["criteria"][0]
    )
    definition["goals"]["subgoal-01"]["criteria"].append(first)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_goal_criteria(definition, state())


def test_unknown_goal_rejected():
    definition = criteria()
    definition["goals"]["missing"] = (
        definition["goals"].pop("subgoal-01")
    )

    with pytest.raises(ValueError, match="Unknown"):
        validate_goal_criteria(definition, state())


def test_evidence_deduplicated():
    quote = "I am Mara, Senior Archivist."
    result = replay_goal_criteria(
        criteria(),
        state(),
        {
            1: event(1, quote),
            2: event(2, quote),
        },
    )

    evidence = result["goals"]["subgoal-01"][
        "criteria"
    ]["identify-mara"]["evidence"]

    assert len(evidence) == 1
