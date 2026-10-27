"""Mixed observable/offscreen knowledge pipeline end to end."""

import json

from rpbench.core import Backend
from rpbench.evaluator_runtime import reconcile_evaluations
from rpbench.knowledge_gate_journal import read_gate_events
from rpbench.observations import read_jsonl


PLAYER = {"known_facts": []}
STATE = {"goals": {}}

SETTINGS = {
    "mode": "llm",
    "id": "mixed-gate-e2e",
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock-evaluator",
    },
    "sampling": {"max_tokens": 512},
    "max_attempts": 1,
}

NPC_TEXT = (
    "Mara tells Elena that the public reading room "
    "closes at six o'clock.\n\n"
    "Elena thanks Mara and leaves the archive.\n\n"
    "After Elena leaves, Voss privately tells Mara "
    "that the vault key is hidden beneath the blue lantern."
)

PLAYER_TEXT = (
    "Elena asks Mara when the reading room closes. "
    "After hearing the answer, she leaves the archive."
)


def test_mixed_scene_filters_offscreen_secret_and_replays(
    tmp_path, monkeypatch,
):
    observations = [
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "player",
            "text": PLAYER_TEXT,
        },
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "npc",
            "text": NPC_TEXT,
        },
    ]

    calls = {
        "gate": 0,
        "knowledge": 0,
    }

    def generate(self, url, payload):
        assert self.model == "mock-evaluator"

        messages = payload["messages"]
        system = messages[0]["content"]
        context = json.loads(messages[1]["content"])

        if "PLAYER-PERSPECTIVE SCENE BOUNDARIES" in system:
            calls["gate"] += 1

            assert context["player_action"] == PLAYER_TEXT
            assert len(context["paragraphs"]) == 3

            content = json.dumps({
                "labels": [
                    {"id": 0, "visibility": "observable"},
                    {"id": 1, "visibility": "observable"},
                    {"id": 2, "visibility": "offscreen"},
                ],
            })

        else:
            calls["knowledge"] += 1

            assert context["source_turn"] == 1
            assert context["latest_player_action"] == PLAYER_TEXT

            visible = context["latest_npc_response"]

            assert "closes at six o'clock" in visible
            assert "Elena thanks Mara" in visible

            # The private secret must never reach the
            # knowledge extraction model.
            assert "vault key" not in visible
            assert "blue lantern" not in visible
            assert "Voss privately tells Mara" not in visible

            assert "known_information" not in context

            content = json.dumps({
                "knowledge_updates": [{
                    "fact": (
                        "Mara tells Elena that the public reading "
                        "room closes at six o'clock."
                    ),
                    "source_turns": [1],
                }],
                "goal_updates": [],
            })

        return content, {
            "choices": [{
                "message": {
                    "content": content,
                    "reasoning_content": "Mock reasoning",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "completion_tokens": 25,
            },
        }

    monkeypatch.setattr(Backend, "generate", generate)

    knowledge = reconcile_evaluations(
        tmp_path,
        PLAYER,
        observations,
        STATE,
        [],
        SETTINGS,
        knowledge_only=True,
    )

    assert calls == {
        "gate": 1,
        "knowledge": 1,
    }

    facts = [
        item["fact"]
        for item in knowledge["facts"]
    ]

    assert len(facts) == 1
    assert "closes at six o'clock" in facts[0]
    assert not any(
        "vault" in fact.lower()
        or "lantern" in fact.lower()
        for fact in facts
    )

    events = read_jsonl(tmp_path / "events.jsonl")

    gate_events = [
        item for item in events
        if item.get("type") == "knowledge_gate_v1"
    ]

    evaluation_events = [
        item for item in events
        if item.get("type") == "player_evaluation_v1"
    ]

    assert len(gate_events) == 1
    assert len(evaluation_events) == 1

    assert gate_events[0]["labels"] == [
        "observable",
        "observable",
        "offscreen",
    ]

    archived = read_gate_events(
        tmp_path / "events.jsonl",
        observations,
    )
    assert archived[1] == gate_events[0]

    # Reconciliation must not perform new inference
    # when both events have already been archived.
    knowledge_again = reconcile_evaluations(
        tmp_path,
        PLAYER,
        observations,
        STATE,
        [],
        SETTINGS,
        knowledge_only=True,
    )

    assert calls == {
        "gate": 1,
        "knowledge": 1,
    }
    assert knowledge_again == knowledge
