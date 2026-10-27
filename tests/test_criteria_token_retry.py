"""Regression tests for criteria generation token escalation."""

import pytest

from rpbench import criteria_runtime
from rpbench.core import dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl

from test_quest_direct_supersession import (
    GROUND_TRUTH,
    DEFINITION,
)


SETTINGS = {
    "id": "criteria-token-retry-test",
    "max_attempts": 2,
    "quest_scheduler": "direct_supersession",
    "sampling": {"temperature": 1.0, "max_tokens": 16384},
}


def setup(root):
    dump(root / "ground_truth.json", GROUND_TRUTH)
    return [
        {
            "turn": 1,
            "role": "player",
            "text": "Elena asks Voss for restricted access.",
        },
        {
            "turn": 1,
            "role": "npc",
            "text": "Voss grants restricted access immediately.",
        },
    ]


def response(goal_id, criterion_ids, status):
    ids = criterion_ids or [
        item["id"]
        for item in DEFINITION["goals"][goal_id]["criteria"]
    ]
    return {
        "schema_version": 1,
        "criterion_updates": [
            {
                "goal_id": goal_id,
                "criterion_id": cid,
                "status": status,
                "source_turns": (
                    [1] if status != "not_satisfied" else []
                ),
                "reason": "Test evidence",
            }
            for cid in ids
        ],
    }


def install(monkeypatch, behavior):
    calls = []

    def fake_generate(
        definition, progress, observations, settings,
        on_response=None, goal_id=None, criterion_ids=None,
        repair=None,
    ):
        budget = settings["sampling"]["max_tokens"]
        calls.append({
            "budget": budget,
            "repair": repair,
            "goal_id": goal_id,
        })

        result = behavior(budget, len(calls))

        if result == "truncated":
            if on_response is not None:
                on_response({
                    "content": "",
                    "reasoning": "unfinished thinking",
                    "finish_reason": "length",
                    "usage": {"completion_tokens": budget},
                })
            raise ValueError("Generation exhausted token budget")

        proposal = response(goal_id, criterion_ids, result)
        metadata = {
            "content": '{"schema_version":1}',
            "reasoning": "",
            "finish_reason": "stop",
            "usage": {"completion_tokens": 200},
        }
        if on_response is not None:
            on_response(metadata)
        return proposal, metadata

    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate
    )
    return calls


def attempts(root):
    return [
        event
        for event in read_jsonl(root / "events.jsonl")
        if event["type"] == "criteria_attempt"
    ]


def test_truncation_retries_once_with_32k(tmp_path, monkeypatch):
    calls = install(
        monkeypatch,
        lambda budget, call: (
            "truncated" if call == 1 else "satisfied"
        ),
    )

    refresh_criteria(tmp_path, DEFINITION, setup(tmp_path), SETTINGS)

    assert [c["budget"] for c in calls] == [16384, 32768]
    assert calls[1]["repair"] is None

    logged = attempts(tmp_path)
    assert [e["status"] for e in logged] == [
        "rejected", "accepted"
    ]
    assert [e["max_tokens"] for e in logged] == [
        16384, 32768
    ]

    accepted = [
        e for e in read_jsonl(tmp_path / "events.jsonl")
        if e["type"] == "semantic_criteria_v1"
    ]
    assert len(accepted) == 1


def test_valid_partial_does_not_retry(tmp_path, monkeypatch):
    calls = install(
        monkeypatch,
        lambda budget, call: "partial",
    )

    refresh_criteria(tmp_path, DEFINITION, setup(tmp_path), SETTINGS)

    assert [c["budget"] for c in calls] == [
        16384, 16384
    ]
    assert [c["goal_id"] for c in calls] == [
        "subgoal-08", "subgoal-07"
    ]
    assert [e["status"] for e in attempts(tmp_path)] == [
        "accepted", "accepted"
    ]


def test_success_first_attempt_does_not_retry(
    tmp_path, monkeypatch,
):
    calls = install(
        monkeypatch,
        lambda budget, call: "satisfied",
    )

    refresh_criteria(tmp_path, DEFINITION, setup(tmp_path), SETTINGS)

    # Direct authorization supersedes the normal request goal.
    assert [c["budget"] for c in calls] == [16384]
    assert [c["goal_id"] for c in calls] == [
        "subgoal-08"
    ]
    assert [e["status"] for e in attempts(tmp_path)] == [
        "accepted"
    ]
    assert [e["max_tokens"] for e in attempts(tmp_path)] == [
        16384
    ]


def test_second_truncation_stops(tmp_path, monkeypatch):
    calls = install(
        monkeypatch,
        lambda budget, call: "truncated",
    )

    with pytest.raises(ValueError, match="token budget"):
        refresh_criteria(
            tmp_path, DEFINITION, setup(tmp_path), SETTINGS
        )

    assert [c["budget"] for c in calls] == [16384, 32768]
    assert [e["status"] for e in attempts(tmp_path)] == [
        "rejected", "rejected"
    ]

    assert not any(
        e["type"] == "semantic_criteria_v1"
        for e in read_jsonl(tmp_path / "events.jsonl")
    )


def test_invalid_json_keeps_original_budget(tmp_path, monkeypatch):
    calls = []

    def fake_generate(
        definition, progress, observations, settings,
        on_response=None, goal_id=None, criterion_ids=None,
        repair=None,
    ):
        budget = settings["sampling"]["max_tokens"]
        calls.append((budget, repair))

        if len(calls) == 1:
            on_response({
                "content": "not valid JSON",
                "reasoning": "",
                "finish_reason": "stop",
                "usage": {"completion_tokens": 10},
            })
            raise ValueError("Invalid criteria JSON")

        return response(goal_id, criterion_ids, "satisfied"), {
            "content": '{"schema_version":1}',
            "reasoning": "",
            "finish_reason": "stop",
            "usage": {"completion_tokens": 100},
        }

    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate
    )

    refresh_criteria(
        tmp_path, DEFINITION, setup(tmp_path), SETTINGS
    )

    assert [budget for budget, _ in calls] == [
        16384, 16384
    ]
    assert calls[0][1] is None
    assert calls[1][1] is not None
    assert "not valid JSON" in calls[1][1]["previous_output"]

    assert [event["status"] for event in attempts(tmp_path)] == [
        "rejected", "accepted"
    ]
