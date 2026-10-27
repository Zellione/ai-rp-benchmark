"""Target exactly one semantic criterion without changing defaults."""

import json

import pytest

from rpbench.core import Backend
from rpbench.criteria_runtime import _generate


DEFINITION = {
    "goals": {
        "subgoal-08": {
            "criteria": [
                {
                    "id": "authorized-access",
                    "description": "Receive restricted authorization",
                },
                {
                    "id": "document-examined",
                    "description": "Examine restricted documents",
                },
            ],
        },
    },
}

PROGRESS = {
    "goals": {
        "subgoal-08": {
            "criteria": {
                "authorized-access": {
                    "proposed_satisfied": False,
                },
                "document-examined": {
                    "proposed_satisfied": False,
                },
            },
        },
    },
}

OBSERVATIONS = [
    {
        "turn": 3,
        "role": "player",
        "text": "Elena asks Voss about access.",
    },
    {
        "turn": 3,
        "role": "npc",
        "text": "Voss grants Elena restricted authorization.",
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


def test_targeted_request_only_contains_selected_criterion(monkeypatch):
    seen = []

    def fake_generate(self, url, payload):
        context = json.loads(payload["messages"][1]["content"])
        seen.append(context["open_criteria"])

        content = json.dumps({
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
                "status": "satisfied",
                "source_turns": [3],
                "reason": "Explicit grant",
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

    result, metadata = _generate(
        DEFINITION,
        PROGRESS,
        OBSERVATIONS,
        SETTINGS,
        goal_id="subgoal-08",
        criterion_ids=["authorized-access"],
    )

    assert seen == [{
        "subgoal-08": [{
            "id": "authorized-access",
            "description": "Receive restricted authorization",
            "prior_partial_evidence": [],
        }],
    }]
    assert result["criterion_updates"][0]["status"] == "satisfied"
    assert metadata["usage"]["completion_tokens"] == 30


def test_default_still_includes_all_open_criteria(monkeypatch):
    def fake_generate(self, url, payload):
        context = json.loads(payload["messages"][1]["content"])
        ids = [
            item["id"]
            for item in context["open_criteria"]["subgoal-08"]
        ]
        assert ids == [
            "authorized-access",
            "document-examined",
        ]

        content = json.dumps({
            "schema_version": 1,
            "criterion_updates": [],
        })
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    _generate(
        DEFINITION, PROGRESS, OBSERVATIONS, SETTINGS,
        goal_id="subgoal-08",
    )


@pytest.mark.parametrize("targets", [
    [],
    ["unknown"],
    ["authorized-access", "authorized-access"],
    [42],
])
def test_invalid_targets_rejected(targets):
    with pytest.raises(ValueError):
        _generate(
            DEFINITION, PROGRESS, OBSERVATIONS, SETTINGS,
            goal_id="subgoal-08",
            criterion_ids=targets,
        )


def test_completed_target_rejected_before_inference():
    progress = {
        "goals": {
            "subgoal-08": {
                "criteria": {
                    **PROGRESS["goals"]["subgoal-08"]["criteria"],
                    "authorized-access": {
                        "proposed_satisfied": True,
                    },
                },
            },
        },
    }

    with pytest.raises(ValueError, match="No open criteria"):
        _generate(
            DEFINITION, progress, OBSERVATIONS, SETTINGS,
            goal_id="subgoal-08",
            criterion_ids=["authorized-access"],
        )
