"""Live criteria request context and failure diagnostics."""

import json

import pytest

from rpbench.core import Backend
from rpbench.criteria_runtime import _generate, refresh_criteria
from rpbench.core import dump
from rpbench.state import initial_state


DEFINITION = {
    "schema_version": 1,
    "goals": {
        "subgoal-01": {
            "criteria": [{
                "id": "mara-role",
                "description": "Learn Mara's responsibility",
                "evidence_contains": "Senior Archivist",
            }],
        },
    },
}


def make_state():
    return {
        "last_confirmed_event": 0,
        "goals": {
            "subgoal-01": {
                "description": "Learn archive staff",
                "status": "pending",
            },
        },
    }


def settings():
    return {
        "mode": "llm",
        "id": "criteria-diagnostic-test",
        "backend": {
            "name": "llamacpp",
            "base_url": "http://127.0.0.1:18081",
            "model": "test-evaluator",
        },
        "sampling": {"max_tokens": 8192},
    }


def observations():
    return [{
        "role": "npc",
        "turn": 1,
        "text": "I am Mara, Senior Archivist.",
    }]


def test_only_current_npc_text_in_request(monkeypatch):
    captured = {}

    def fake_generate(self, url, payload):
        context = json.loads(
            payload["messages"][1]["content"]
        )
        captured.update(context)

        output = json.dumps({
            "schema_version": 1,
            "criterion_updates": [],
        })
        return output, {
            "choices": [{
                "message": {"content": output},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    progress = {
        "goals": {
            "subgoal-01": {
                "criteria": {
                    "mara-role": {
                        "description": "Learn Mara's responsibility",
                        "proposed_satisfied": False,
                        "evidence": [],
                    },
                },
                "proposed_satisfied": False,
            },
        },
    }

    entries = observations() + [{
        "role": "npc",
        "turn": 2,
        "text": "This is a new conversation turn.",
    }]

    _generate(DEFINITION, progress, entries, settings())

    assert captured["latest_npc_response"]["turn"] == 2
    assert "Senior Archivist" not in json.dumps(
        captured["latest_npc_response"]
    )
    assert "npc_dialogue" not in captured


def test_failure_archives_raw_thinking(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()

    ground_truth = {
        "schema_version": 1,
        "player": {"subgoals": {}},
        "characters": {},
    }
    dump(root / "ground_truth.json", ground_truth)

    # Use the real state builder's structure but install the
    # one synthetic goal used by this test.
    state = initial_state(ground_truth)
    state["goals"] = make_state()["goals"]

    ground_truth["player"]["subgoals"] = {
        "subgoal-01": {
            "description": "Learn archive staff",
            "initial_status": "pending",
        },
    }
    dump(root / "ground_truth.json", ground_truth)

    def fake_generate(self, url, payload):
        return "", {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "Long model reasoning",
                },
                "finish_reason": "length",
            }],
            "usage": {"completion_tokens": 8192},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    with pytest.raises(ValueError, match="Invalid criteria JSON"):
        refresh_criteria(
            root, DEFINITION, observations(), settings()
        )

    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1

    event = events[0]
    assert event["status"] == "rejected"
    assert event["raw_response"]["reasoning"] == (
        "Long model reasoning"
    )
    assert event["raw_response"]["finish_reason"] == "length"
    assert event["raw_response"]["usage"]["completion_tokens"] == 8192
    assert not (root / "criteria_progress.json").exists()


def test_evaluator_keeps_later_goals_available(monkeypatch):
    captured = {}

    def fake_generate(self, url, payload):
        captured.update(json.loads(
            payload["messages"][1]["content"]
        ))
        text = json.dumps({
            "schema_version": 1,
            "criterion_updates": [],
        })
        return text, {
            "choices": [{
                "message": {"content": text},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [{
                    "id": "first",
                    "description": "First criterion",
                    "evidence_contains": "first",
                }],
            },
            "subgoal-02": {
                "criteria": [{
                    "id": "second",
                    "description": "Second criterion",
                    "evidence_contains": "second",
                }],
            },
        },
    }

    progress = {
        "goals": {
            goal_id: {
                "criteria": {
                    criterion_id: {
                        "description": criterion_id,
                        "proposed_satisfied": False,
                        "evidence": [],
                    },
                },
                "proposed_satisfied": False,
            }
            for goal_id, criterion_id in (
                ("subgoal-01", "first"),
                ("subgoal-02", "second"),
            )
        },
    }

    _generate(
        definition, progress,
        observations(), settings(),
        goal_id="subgoal-01",
    )

    assert captured["active_goal_id"] == "subgoal-01"
    assert set(captured["open_criteria"]) == {"subgoal-01"}

    _generate(
        definition, progress,
        observations(), settings(),
        goal_id="subgoal-02",
    )

    assert captured["active_goal_id"] == "subgoal-02"
    assert set(captured["open_criteria"]) == {"subgoal-02"}
