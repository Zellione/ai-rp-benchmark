"""Gate inherits the evaluator generation budget."""

import json

import pytest

from rpbench.core import Backend
from rpbench.knowledge_gate import evaluate_gate


SETTINGS = {
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock",
    },
    "sampling": {
        "temperature": 1.0,
        "max_tokens": 32768,
    },
}


def test_gate_inherits_evaluator_budget(monkeypatch):
    seen = []

    def fake_generate(self, url, payload):
        seen.append(payload["max_tokens"])

        content = json.dumps({
            "labels": [{
                "id": 0,
                "visibility": "observable",
            }],
        })

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 30},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    labels, _ = evaluate_gate(
        "Elena enters the archive.",
        "Mara welcomes Elena.",
        SETTINGS,
    )

    assert labels == ["observable"]
    assert seen == [32768]


def test_explicit_gate_budget(monkeypatch):
    seen = []

    def fake_generate(self, url, payload):
        seen.append(payload["max_tokens"])
        content = '{"labels":[{"id":0,"visibility":"observable"}]}'
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    evaluate_gate(
        "Elena listens.",
        "Mara speaks.",
        SETTINGS,
        max_tokens=8192,
    )

    assert seen == [8192]


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "4096", 131073])
def test_invalid_gate_budget_rejected(invalid):
    with pytest.raises(ValueError, match="token budget"):
        evaluate_gate(
            "Elena listens.",
            "Mara speaks.",
            SETTINGS,
            max_tokens=invalid,
        )
