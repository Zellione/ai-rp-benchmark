from copy import deepcopy

import pytest

from rpbench.false_positive_branch import (
    rejected_criteria,
    correct_semantic_events,
)
from rpbench.semantic_criteria import make_semantic_event


def verdict(a="satisfied", b="partial"):
    return {
        "criteria": [
            {"criterion_id": "a", "status": a},
            {"criterion_id": "b", "status": b},
        ]
    }


def test_reject_only_unfulfilled_criteria():
    assert rejected_criteria(verdict()) == {"b"}


def test_no_rejected_criteria_is_not_a_rollback():
    with pytest.raises(ValueError, match="no disproven"):
        rejected_criteria(verdict("satisfied", "satisfied"))


def test_uncertain_audit_blocks_rollback():
    with pytest.raises(ValueError, match="Uncertain"):
        rejected_criteria(verdict("satisfied", "uncertain"))


def test_rewrite_preserves_unrelated_events():
    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [
                    {
                        "id": "a",
                        "description": "A",
                        "evidence_contains": "a",
                    },
                    {
                        "id": "b",
                        "description": "B",
                        "evidence_contains": "b",
                    },
                ],
            },
        },
    }

    state = {
        "goals": {
            "subgoal-01": {
                "description": "Test",
                "status": "pending",
            },
        },
    }

    observations = [
        {
            "turn": 1,
            "role": "npc",
            "text": "Evidence A and B",
        },
    ]

    proposal = {
        "schema_version": 1,
        "criterion_updates": [
            {
                "goal_id": "subgoal-01",
                "criterion_id": cid,
                "status": "satisfied",
                "source_turns": [1],
                "reason": f"Evidence {cid}",
            }
            for cid in ("a", "b")
        ],
    }

    event = make_semantic_event(
        1, proposal, definition, state,
        observations, "test-judge",
        goal_id="subgoal-01",
    )

    metadata = {"type": "request", "turn": 1, "role": "npc"}
    original = [metadata, event]
    snapshot = deepcopy(original)

    corrected, changes = correct_semantic_events(
        original,
        definition,
        state,
        observations,
        1,
        "subgoal-01",
        {"b"},
    )

    assert original == snapshot
    assert corrected[0] == metadata
    assert changes[0]["removed_criteria"] == ["b"]

    remaining = corrected[1]["result"]["criterion_updates"]

    assert [item["criterion_id"] for item in remaining] == ["a"]
    assert corrected[1] != event


def test_unknown_criterion_is_rejected():
    definition = {
        "goals": {
            "subgoal-01": {
                "criteria": [{"id": "a"}],
            },
        },
    }

    with pytest.raises(ValueError, match="Unknown"):
        correct_semantic_events(
            [], definition, {}, [], 1,
            "subgoal-01", {"unknown"},
        )
