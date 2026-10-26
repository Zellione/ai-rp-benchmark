"""Provenance and deterministic replay of semantic criteria."""

from copy import deepcopy

import pytest

from rpbench.semantic_criteria import (
    make_semantic_event,
    replay_semantic_criteria,
    validate_semantic_proposal,
)


def state():
    return {
        "goals": {
            "subgoal-01": {
                "description": "Identify archive staff",
                "status": "pending",
            },
        },
        "last_confirmed_event": 0,
    }


def definition():
    return {
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


def observations():
    return [
        {
            "turn": 1,
            "role": "npc",
            "text": "I am Mara, Senior Archivist.",
        },
        {
            "turn": 2,
            "role": "player",
            "text": "And Finn?",
        },
        {
            "turn": 3,
            "role": "npc",
            "text": "Finn handles digitization.",
        },
    ]


def proposal(criterion, turn, quote):
    return {
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


def test_multiturn_semantic_progress():
    first = make_semantic_event(
        1,
        proposal("mara", 1, "Senior Archivist"),
        definition(), state(), observations(), "evaluator-v2",
    )
    second = make_semantic_event(
        3,
        proposal("finn", 3, "Finn handles digitization."),
        definition(), state(), observations(), "evaluator-v2",
    )

    initial = state()
    original = deepcopy(initial)

    progress = replay_semantic_criteria(
        definition(), initial, observations(),
        [first, second],
    )

    goal = progress["goals"]["subgoal-01"]
    assert goal["proposed_satisfied"]
    assert goal["criteria"]["mara"]["evidence"][0]["turn"] == 1
    assert goal["criteria"]["finn"]["evidence"][0]["turn"] == 3

    assert initial == original
    assert initial["goals"]["subgoal-01"]["status"] == "pending"


def test_future_evidence_rejected():
    with pytest.raises(ValueError, match="source mismatch"):
        validate_semantic_proposal(
            proposal("finn", 3, "Finn handles digitization."),
            definition(), state(), observations(), 1,
        )


def test_fabricated_quote_rejected():
    with pytest.raises(ValueError, match="source mismatch"):
        validate_semantic_proposal(
            proposal("mara", 1, "Mara is the director."),
            definition(), state(), observations(), 1,
        )


def test_unknown_criterion_rejected():
    with pytest.raises(ValueError, match="criterion"):
        validate_semantic_proposal(
            proposal("unknown", 1, "Senior Archivist"),
            definition(), state(), observations(), 1,
        )


def test_duplicate_criterion_rejected():
    data = proposal("mara", 1, "Senior Archivist")
    data["criterion_updates"].append(
        deepcopy(data["criterion_updates"][0])
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_semantic_proposal(
            data, definition(), state(), observations(), 1,
        )


def test_tampered_event_rejected():
    event = make_semantic_event(
        1,
        proposal("mara", 1, "Senior Archivist"),
        definition(), state(), observations(), "evaluator-v2",
    )
    event["definition_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="provenance"):
        replay_semantic_criteria(
            definition(), state(), observations(), [event],
        )


def test_conflicting_archived_turn_rejected():
    event = make_semantic_event(
        1,
        proposal("mara", 1, "Senior Archivist"),
        definition(), state(), observations(), "evaluator-v2",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        replay_semantic_criteria(
            definition(), state(), observations(),
            [event, deepcopy(event)],
        )


def test_invalid_status_rejected():
    data = proposal("mara", 1, "Senior Archivist")
    data["criterion_updates"][0]["status"] = "failed"

    with pytest.raises(ValueError, match="status"):
        validate_semantic_proposal(
            data, definition(), state(), observations(), 1,
        )


def test_multiple_quotes_across_turns():
    data = proposal("mara", 1, "Senior Archivist")
    data["criterion_updates"][0]["evidence"].append({
        "turn": 3,
        "quote": "Finn handles digitization.",
    })

    validated = validate_semantic_proposal(
        data, definition(), state(), observations(), 3,
    )

    assert len(
        validated["criterion_updates"][0]["evidence"]
    ) == 2
