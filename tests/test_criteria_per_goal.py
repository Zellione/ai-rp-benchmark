"""Independent per-goal evaluation and partial resume."""

import json

import pytest

from rpbench.core import Backend, dump, load
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl


def test_partial_resume_preserves_checked_goals(
    tmp_path, monkeypatch,
):
    root = tmp_path / "run"
    root.mkdir()

    ground_truth = {
        "schema_version": 1,
        "player": {
            "subgoals": {
                "subgoal-01": {
                    "description": "First goal",
                    "initial_status": "pending",
                },
                "subgoal-02": {
                    "description": "Second goal",
                    "initial_status": "pending",
                },
            },
        },
        "characters": {},
    }
    dump(root / "ground_truth.json", ground_truth)

    definition = {
        "schema_version": 1,
        "goals": {
            goal_id: {
                "criteria": [{
                    "id": criterion_id,
                    "description": description,
                    "evidence_contains": criterion_id,
                }],
            }
            for goal_id, criterion_id, description in (
                ("subgoal-01", "first", "Learn the first fact"),
                ("subgoal-02", "second", "Learn the second fact"),
            )
        },
    }

    observations = [{
        "role": "player",
        "turn": 1,
        "text": "Tell me about both subjects.",
    }, {
        "role": "npc",
        "turn": 1,
        "text": "The first fact is here. The second fact is here.",
    }]

    settings = {
        "mode": "llm",
        "id": "per-goal-test",
        "backend": {
            "name": "llamacpp",
            "base_url": "http://127.0.0.1:18081",
            "model": "test-evaluator",
        },
        "sampling": {"max_tokens": 512},
        "max_attempts": 1,
    }

    calls = []
    should_fail = {"second": True}

    def generate(self, url, payload):
        context = json.loads(
            payload["messages"][1]["content"]
        )
        assert context["latest_player_action"] == (
            "Tell me about both subjects."
        )

        goal_id = next(iter(context["open_criteria"]))
        calls.append(goal_id)

        if goal_id == "subgoal-02" and should_fail["second"]:
            return "", {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": "unfinished thinking",
                    },
                    "finish_reason": "length",
                }],
            }

        criterion_id = (
            "first" if goal_id == "subgoal-01" else "second"
        )
        quote = (
            "The first fact is here."
            if criterion_id == "first"
            else "The second fact is here."
        )

        content = json.dumps({
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": goal_id,
                "criterion_id": criterion_id,
                "status": "satisfied",
                "evidence": [{"turn": 1, "quote": quote}],
            }],
        })
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    with pytest.raises(ValueError, match="Invalid criteria JSON"):
        refresh_criteria(
            root, definition, observations, settings
        )

    assert calls == ["subgoal-01", "subgoal-02"]

    journal = read_jsonl(root / "events.jsonl")
    accepted = [
        e for e in journal
        if e.get("type") == "semantic_criteria_v1"
    ]
    assert len(accepted) == 1
    assert accepted[0]["goal_id"] == "subgoal-01"

    should_fail["second"] = False

    state, progress, _ = refresh_criteria(
        root, definition, observations, settings
    )

    assert calls == [
        "subgoal-01", "subgoal-02", "subgoal-02"
    ]
    assert all(
        g["status"] == "completed"
        for g in state["goals"].values()
    )
    assert all(
        g["proposed_satisfied"]
        for g in progress["goals"].values()
    )

    refresh_criteria(
        root, definition, observations, settings
    )
    assert len(calls) == 3


def test_later_goal_can_complete_before_first(
    tmp_path, monkeypatch,
):
    root = tmp_path / "run"
    root.mkdir()

    dump(root / "ground_truth.json", {
        "schema_version": 1,
        "player": {
            "subgoals": {
                "subgoal-01": {
                    "description": "First goal",
                    "initial_status": "pending",
                },
                "subgoal-02": {
                    "description": "Second goal",
                    "initial_status": "pending",
                },
            },
        },
        "characters": {},
    })

    definition = {
        "schema_version": 1,
        "goals": {
            goal_id: {
                "criteria": [{
                    "id": criterion_id,
                    "description": criterion_id,
                    "evidence_contains": criterion_id,
                }],
            }
            for goal_id, criterion_id in (
                ("subgoal-01", "first"),
                ("subgoal-02", "second"),
            )
        },
    }

    observations = [
        {
            "role": "player",
            "turn": 1,
            "text": "What have we discovered?",
        },
        {
            "role": "npc",
            "turn": 1,
            "text": "We found the second fact.",
        },
    ]

    settings = {
        "mode": "llm",
        "id": "out-of-order-test",
        "backend": {
            "name": "llamacpp",
            "base_url": "http://127.0.0.1:18081",
            "model": "test-evaluator",
        },
        "sampling": {"max_tokens": 512},
    }

    def generate(self, url, payload):
        context = json.loads(
            payload["messages"][1]["content"]
        )
        goal_id = next(iter(context["open_criteria"]))

        updates = []
        if goal_id == "subgoal-02":
            updates = [{
                "goal_id": "subgoal-02",
                "criterion_id": "second",
                "status": "satisfied",
                "evidence": [{
                    "turn": 1,
                    "quote": "We found the second fact.",
                }],
            }]

        content = json.dumps({
            "schema_version": 1,
            "criterion_updates": updates,
        })

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    state, progress, transitions = refresh_criteria(
        root, definition, observations, settings
    )

    assert state["goals"]["subgoal-01"]["status"] == "pending"
    assert state["goals"]["subgoal-02"]["status"] == "completed"
    assert not progress["goals"]["subgoal-01"]["proposed_satisfied"]
    assert progress["goals"]["subgoal-02"]["proposed_satisfied"]

    assert len(transitions) == 1
    assert transitions[0]["goal_id"] == "subgoal-02"

    # The controlled player should continue pursuing the
    # earliest still-open goal, rather than redoing goal 02.
    from rpbench.player_controller import select_player_goal

    assert select_player_goal(state)["id"] == "subgoal-01"

    events = read_jsonl(root / "events.jsonl")
    assessed = [
        e for e in events
        if e.get("type") == "semantic_criteria_v1"
    ]

    assert len(assessed) == 2
    assert {e["goal_id"] for e in assessed} == {
        "subgoal-01", "subgoal-02"
    }
