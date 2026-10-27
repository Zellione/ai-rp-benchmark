"""Criteria format recovery without weakening provenance checks."""

import json

from rpbench.core import Backend, dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl


GROUND_TRUTH = {
    "schema_version": 1,
    "player": {
        "subgoals": {
            "subgoal-01": {
                "description": "Obtain restricted authorization",
                "initial_status": "pending",
            },
        },
    },
    "characters": {},
}

DEFINITION = {
    "schema_version": 1,
    "goals": {
        "subgoal-01": {
            "criteria": [{
                "id": "authorized-access",
                "description": "Receive actual restricted access",
                "evidence_contains": "authorization",
            }],
        },
    },
}

SETTINGS = {
    "id": "criteria-format-repair",
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock",
    },
    "sampling": {"max_tokens": 1024},
    "max_attempts": 2,
}


def observations(npc_text):
    return [
        {
            "turn": 1,
            "role": "player",
            "text": "Elena asks for permission.",
        },
        {
            "turn": 1,
            "role": "npc",
            "text": npc_text,
        },
    ]


def install_backend(monkeypatch, responses, seen):
    def fake_generate(self, url, payload):
        seen.append(payload["messages"])
        content = responses[len(seen) - 1]
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 20},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)


def test_prose_response_repaired_and_reused_on_resume(
    tmp_path, monkeypatch,
):
    dump(tmp_path / "ground_truth.json", GROUND_TRUTH)
    source = observations(
        "Mara hands Elena the application form. "
        "No authorization has been granted."
    )

    valid = json.dumps({
        "schema_version": 1,
        "criterion_updates": [{
            "goal_id": "subgoal-01",
            "criterion_id": "authorized-access",
            "status": "not_satisfied",
            "source_turns": [],
            "reason": "An application form is not authorization.",
        }],
    })

    seen = []
    install_backend(
        monkeypatch,
        [
            "Authorization is not satisfied: only a form was given.",
            valid,
        ],
        seen,
    )

    state, _, _ = refresh_criteria(
        tmp_path, DEFINITION, source, SETTINGS
    )
    assert state["goals"]["subgoal-01"]["status"] == "pending"
    assert len(seen) == 2
    assert len(seen[0]) == 2
    assert len(seen[1]) == 3
    assert "ONLY the required JSON" in seen[1][-1]["content"]

    events = read_jsonl(tmp_path / "events.jsonl")
    attempts = [
        event for event in events
        if event["type"] == "criteria_attempt"
    ]
    assert [event["status"] for event in attempts] == [
        "rejected", "accepted"
    ]
    assert attempts[0]["raw_response"]["content"].startswith(
        "Authorization is not satisfied"
    )

    refresh_criteria(tmp_path, DEFINITION, source, SETTINGS)
    assert len(seen) == 2
    assert len(read_jsonl(tmp_path / "events.jsonl")) == len(events)


def test_invalid_evidence_cannot_be_accepted_by_repair(
    tmp_path, monkeypatch,
):
    dump(tmp_path / "ground_truth.json", GROUND_TRUTH)
    source = observations("Mara shows Elena a public reading pass.")

    invalid = json.dumps({
        "schema_version": 1,
        "criterion_updates": [{
            "goal_id": "subgoal-01",
            "criterion_id": "authorized-access",
            "status": "satisfied",
            "source_turns": [99],
            "reason": "Invented authorization.",
        }],
    })

    corrected = json.dumps({
        "schema_version": 1,
        "criterion_updates": [{
            "goal_id": "subgoal-01",
            "criterion_id": "authorized-access",
            "status": "not_satisfied",
            "source_turns": [],
            "reason": "A public reading pass is not authorization.",
        }],
    })

    seen = []
    install_backend(monkeypatch, [invalid, corrected], seen)

    state, _, _ = refresh_criteria(
        tmp_path, DEFINITION, source, SETTINGS
    )
    assert state["goals"]["subgoal-01"]["status"] == "pending"
    assert len(seen) == 2

    events = read_jsonl(tmp_path / "events.jsonl")
    accepted = [
        event for event in events
        if event["type"] == "semantic_criteria_v1"
    ]
    assert len(accepted) == 1
    assert accepted[0]["result"]["criterion_updates"][0][
        "status"
    ] == "not_satisfied"
