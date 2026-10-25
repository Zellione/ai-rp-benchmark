"""Evaluation provenance and replay tests."""

from copy import deepcopy

import pytest

from rpbench.evaluation_journal import (
    append_evaluation,
    read_evaluations,
)


NPC = "Director Vale manages the restricted collection."

STATE = {
    "goals": {
        "subgoal-01": {
            "description": "Identify archive staff",
            "status": "pending",
        },
    },
}

OBSERVATIONS = [{
    "turn": 1,
    "role": "npc",
    "text": NPC,
}]

PROPOSAL = {
    "knowledge_updates": [{
        "fact": "Director Vale manages restricted records.",
        "evidence": NPC,
    }],
    "goal_updates": [{
        "goal_id": "subgoal-01",
        "status": "completed",
        "evidence": NPC,
    }],
}


def test_record_and_replay(tmp_path):
    path = tmp_path / "events.jsonl"

    event = append_evaluation(
        path, 1, NPC, STATE, PROPOSAL,
        "mock-evaluator-v1", OBSERVATIONS,
    )

    restored = read_evaluations(
        path, OBSERVATIONS, STATE
    )

    assert restored[1] == event
    assert event["result"]["knowledge_updates"][0][
        "status"
    ] == "reported"


def test_duplicate_evaluation_rejected(tmp_path):
    path = tmp_path / "events.jsonl"

    args = (
        path, 1, NPC, STATE, PROPOSAL,
        "mock-evaluator-v1", OBSERVATIONS,
    )

    append_evaluation(*args)

    with pytest.raises(ValueError, match="already archived"):
        append_evaluation(*args)


def test_modified_npc_response_rejected(tmp_path):
    path = tmp_path / "events.jsonl"

    append_evaluation(
        path, 1, NPC, STATE, PROPOSAL,
        "mock-evaluator-v1", OBSERVATIONS,
    )

    changed = deepcopy(OBSERVATIONS)
    changed[0]["text"] = "Different response"

    with pytest.raises(ValueError, match="hash mismatch"):
        read_evaluations(path, changed, STATE)


def test_hallucinated_evidence_not_archived(tmp_path):
    path = tmp_path / "events.jsonl"
    proposal = deepcopy(PROPOSAL)
    proposal["knowledge_updates"][0]["evidence"] = (
        "Something the NPC never said"
    )

    with pytest.raises(ValueError):
        append_evaluation(
            path, 1, NPC, STATE, proposal,
            "mock-evaluator-v1", OBSERVATIONS,
        )

    assert not path.exists()


def test_missing_observation_rejected(tmp_path):
    path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match="observation mismatch"):
        append_evaluation(
            path, 1, NPC, STATE, PROPOSAL,
            "mock-evaluator-v1", [],
        )

    assert not path.exists()
