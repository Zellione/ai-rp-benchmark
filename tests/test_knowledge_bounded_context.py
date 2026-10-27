"""Knowledge extraction must remain bounded to the current turn."""

import json

from rpbench.core import Backend
from rpbench.evaluator_llm import evaluate_llm


def _settings():
    return {
        "backend": {
            "name": "llamacpp",
            "base_url": "http://127.0.0.1:18081",
            "model": "test-model",
        },
        "sampling": {"max_tokens": 256},
        "_source_turn": 16,
        "_player_action": "Elena is outside the archive.",
    }


def test_knowledge_extraction_excludes_history(monkeypatch):
    captured = {}

    def generate(self, url, payload):
        captured["messages"] = payload["messages"]
        content = json.dumps({
            "knowledge_updates": [],
            "goal_updates": [],
        })
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    result = evaluate_llm(
        "Mara and Voss speak privately inside the archive.",
        {"goals": {}},
        {"facts": [{
            "fact": "Previously known private information",
            "status": "reported",
        }]},
        _settings(),
        knowledge_only=True,
    )

    context = json.loads(captured["messages"][1]["content"])
    system = captured["messages"][0]["content"]

    assert "known_information" not in context
    assert "Previously known private information" not in (
        json.dumps(captured["messages"])
    )
    assert context["source_turn"] == 16
    assert context["latest_player_action"] == (
        "Elena is outside the archive."
    )
    assert "Mara and Voss speak privately" in (
        context["latest_npc_response"]
    )
    assert "private conversations" in system
    assert result == {
        "knowledge_updates": [],
        "goal_updates": [],
    }


def test_legacy_evaluator_keeps_history(monkeypatch):
    captured = {}

    def generate(self, url, payload):
        captured["context"] = json.loads(
            payload["messages"][1]["content"]
        )
        content = json.dumps({
            "knowledge_updates": [],
            "goal_updates": [],
        })
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    evaluate_llm(
        "NPC response",
        {"goals": {}},
        {"facts": [{
            "fact": "Existing fact",
            "status": "reported",
        }]},
        _settings(),
        knowledge_only=False,
    )

    assert captured["context"]["known_information"] == [{
        "fact": "Existing fact",
        "status": "reported",
    }]
