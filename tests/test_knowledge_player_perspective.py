"""Perspective-aware semantic knowledge prompt."""

import json

from rpbench.core import Backend
from rpbench.evaluator_llm import evaluate_llm


def test_knowledge_prompt_contains_player_action(monkeypatch):
    captured = {}

    def generate(self, url, payload):
        messages = payload["messages"]
        captured["system"] = messages[0]["content"]
        captured["context"] = json.loads(messages[1]["content"])

        content = json.dumps({
            "knowledge_updates": [],
            "goal_updates": [],
        })

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 20},
        }

    monkeypatch.setattr(Backend, "generate", generate)

    evaluate_llm(
        "Mara and Voss talk privately inside the archive.",
        {"goals": {}},
        {"facts": []},
        {
            "backend": {
                "name": "llamacpp",
                "base_url": "http://127.0.0.1:18081",
                "model": "mock",
            },
            "sampling": {"max_tokens": 256},
            "_source_turn": 16,
            "_player_action": (
                "Elena leaves the archive and writes an "
                "email in her car."
            ),
        },
        knowledge_only=True,
    )

    context = captured["context"]

    assert context["source_turn"] == 16
    assert "Elena leaves the archive" in (
        context["latest_player_action"]
    )
    assert "Mara and Voss talk privately" in (
        context["latest_npc_response"]
    )

    instructions = captured["system"]

    assert "private conversations" in instructions
    assert "player's perspective" in instructions
    assert "omit it" in instructions
