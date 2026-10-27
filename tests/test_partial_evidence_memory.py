"""Cumulative partial evidence across independent NPC turns."""

import json

from rpbench.core import Backend, dump
from rpbench.criteria_runtime import _generate, refresh_criteria
from rpbench.observations import read_jsonl
from rpbench.partial_evidence import prior_partial_evidence


GROUND_TRUTH = {
    "schema_version": 1,
    "player": {
        "subgoals": {
            "subgoal-01": {
                "description": "Learn archive staff responsibilities",
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
                "id": "voss-role",
                "description": (
                    "Learn that Alden Voss is the curator "
                    "responsible for access authorization."
                ),
                "evidence_contains": "Voss",
            }],
        },
    },
}

OBSERVATIONS = [
    {
        "turn": 1,
        "role": "player",
        "text": "Elena asks who approves restricted access.",
    },
    {
        "turn": 1,
        "role": "npc",
        "text": (
            "Mara says Mr. Voss is the curator and "
            "authorizes restricted access."
        ),
    },
    {
        "turn": 2,
        "role": "player",
        "text": "Elena waits at reception.",
    },
    {
        "turn": 2,
        "role": "npc",
        "text": "Finn offers Elena a chair.",
    },
    {
        "turn": 3,
        "role": "player",
        "text": "Elena enters the office.",
    },
    {
        "turn": 3,
        "role": "npc",
        "text": (
            "Mara introduces Mr. Voss. "
            "The man at the desk is Alden Voss."
        ),
    },
]

SETTINGS = {
    "id": "partial-memory-test",
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock",
    },
    "sampling": {"max_tokens": 1024},
    "max_attempts": 1,
}


def test_cumulative_voss_evidence_and_resume(
    tmp_path, monkeypatch,
):
    dump(tmp_path / "ground_truth.json", GROUND_TRUTH)
    seen = []

    def fake_generate(self, url, payload):
        context = json.loads(payload["messages"][1]["content"])
        turn = context["latest_npc_response"]["turn"]
        criterion = context["open_criteria"]["subgoal-01"][0]
        previous = criterion["prior_partial_evidence"]
        seen.append((turn, previous))

        if turn == 1:
            assert previous == []
            status, sources = "partial", [1]
            reason = (
                "Mr. Voss is the curator and grants access, "
                "but Alden's full name is not yet established."
            )
        elif turn == 2:
            assert len(previous) == 1
            assert previous[0]["turn"] == 1
            status, sources = "not_satisfied", []
            reason = "No additional evidence."
        else:
            assert len(previous) == 1
            assert previous[0]["turn"] == 1
            assert "curator" in previous[0]["npc_response"]
            status, sources = "satisfied", [1, 3]
            reason = (
                "Turn 1 establishes the curator's responsibility; "
                "turn 3 identifies him as Alden Voss."
            )

        content = json.dumps({
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-01",
                "criterion_id": "voss-role",
                "status": status,
                "source_turns": sources,
                "reason": reason,
            }],
        })

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 20},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    state, progress, _ = refresh_criteria(
        tmp_path, DEFINITION, OBSERVATIONS, SETTINGS
    )

    assert state["goals"]["subgoal-01"]["status"] == "completed"
    assert progress["goals"]["subgoal-01"][
        "criteria"
    ]["voss-role"]["proposed_satisfied"]

    assert [turn for turn, _ in seen] == [1, 2, 3]

    events = read_jsonl(tmp_path / "events.jsonl")
    semantic = [
        event for event in events
        if event["type"] == "semantic_criteria_v1"
    ]

    assert len(semantic) == 3
    final = semantic[-1]["result"]["criterion_updates"][0]
    assert final["source_turns"] == [1, 3]
    assert final["status"] == "satisfied"

    # Archived evaluations are reused on resume.
    refresh_criteria(
        tmp_path, DEFINITION, OBSERVATIONS, SETTINGS
    )

    assert len(seen) == 3
    assert len(read_jsonl(tmp_path / "events.jsonl")) == len(events)


def test_partial_sources_are_bounded_and_historical():
    progress = {
        "evidence": [
            {
                "turn": 1,
                "status": "partial",
                "reason": "First component",
            },
            {
                "turn": 2,
                "status": "not_satisfied",
                "reason": "No component",
            },
            {
                "turn": 3,
                "status": "partial",
                "reason": "Second component",
            },
        ],
    }

    result = prior_partial_evidence(
        progress, OBSERVATIONS, current_turn=3, limit=2
    )

    assert [item["turn"] for item in result] == [1]
    assert result[0]["prior_assessment"] == "First component"
    assert "curator" in result[0]["npc_response"]
