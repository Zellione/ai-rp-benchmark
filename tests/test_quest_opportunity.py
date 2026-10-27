"""Opportunity screening validates targets without awarding goals."""

import json

import pytest

from rpbench.core import Backend
from rpbench.quest_opportunity import (
    check_opportunities,
    parse_opportunities,
)


CANDIDATES = [{
    "goal_id": "subgoal-08",
    "criterion_ids": [
        "authorized-access",
        "document-examined",
    ],
}]

DEFINITION = {
    "goals": {
        "subgoal-08": {
            "criteria": [{
                "id": "authorized-access",
                "description": "Obtain formal authorization",
            }, {
                "id": "document-examined",
                "description": "Examine restricted records",
            }],
        },
    },
}

OBSERVATIONS = [
    {
        "role": "player",
        "turn": 3,
        "text": "Elena asks Voss for help.",
    },
    {
        "role": "npc",
        "turn": 3,
        "text": "Voss grants Elena restricted access.",
    },
]

SETTINGS = {
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock",
    },
    "sampling": {"max_tokens": 1024},
}


def test_spontaneous_authorization_detected(monkeypatch):
    calls = []

    def generate(self, url, payload):
        context = json.loads(payload["messages"][1]["content"])
        calls.append(context)

        content = json.dumps({
            "possible": [{
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
            }],
        })

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 30},
        }

    monkeypatch.setattr(Backend, "generate", generate)

    result, metadata = check_opportunities(
        DEFINITION, CANDIDATES, OBSERVATIONS, SETTINGS
    )

    assert result == [{
        "goal_id": "subgoal-08",
        "criterion_id": "authorized-access",
    }]
    assert len(calls) == 1
    assert calls[0]["latest_npc_response"] == (
        "Voss grants Elena restricted access."
    )
    assert metadata["usage"]["completion_tokens"] == 30


def test_no_candidates_requires_no_model_call():
    result, metadata = check_opportunities(
        DEFINITION, [], OBSERVATIONS, SETTINGS
    )
    assert result == []
    assert metadata is None


@pytest.mark.parametrize("response", [
    '{"possible":[{"goal_id":"unknown","criterion_id":"x"}]}',
    '{"possible":[{"goal_id":"subgoal-08","criterion_id":"unknown"}]}',
    '{"possible":[{"goal_id":"subgoal-08","criterion_id":"authorized-access"},{"goal_id":"subgoal-08","criterion_id":"authorized-access"}]}',
    '{"possible":"authorized-access"}',
    '{"wrong":[]}',
])
def test_invalid_opportunities_rejected(response):
    with pytest.raises(ValueError):
        parse_opportunities(response, CANDIDATES)


def test_empty_opportunity_result():
    assert parse_opportunities(
        '{"possible":[]}', CANDIDATES
    ) == []
