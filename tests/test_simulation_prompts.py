"""Tests for the optional simulation prompt adapter."""

import copy

import pytest

from rpbench.prompt_presets import SillyTavernChatPreset
from rpbench.simulation_prompts import render_npc_chat_messages


def make_preset():
    return SillyTavernChatPreset({
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "You are {{char}}.",
            },
            {
                "identifier": "note",
                "role": "system",
                "content": "Remember the secret.",
                "injection_position": 1,
                "injection_depth": 1,
                "injection_order": 100,
            },
            {
                "identifier": "chatHistory",
                "role": "system",
                "marker": True,
            },
        ],
        "prompt_order": [{
            "character_id": 100000,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "note", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    })


def make_messages():
    return [
        {"role": "system", "content": "Legacy NPC prompt"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Greetings"},
        {"role": "user", "content": "What happened?"},
    ]


def render(messages):
    return render_npc_chat_messages(
        messages,
        make_preset(),
        {},
        character_name="Elara",
        user_name="Rowan",
    )


def test_adapter_uses_preset_instead_of_legacy_system():
    result = render(make_messages())

    assert result == [
        {"role": "system", "content": "You are Elara."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Greetings"},
        {"role": "system", "content": "Remember the secret."},
        {"role": "user", "content": "What happened?"},
    ]


def test_adapter_does_not_mutate_history():
    messages = make_messages()
    original = copy.deepcopy(messages)

    render(messages)

    assert messages == original


def test_adapter_rejects_missing_system_prompt():
    with pytest.raises(ValueError):
        render([{"role": "user", "content": "Hello"}])


def test_adapter_rejects_extra_message_fields():
    messages = make_messages()
    messages[1]["name"] = "Rowan"

    with pytest.raises(ValueError):
        render(messages)


def test_adapter_supports_empty_conversation():
    result = render([
        {"role": "system", "content": "Legacy NPC prompt"}
    ])

    assert result == [
        {"role": "system", "content": "You are Elara."},
        {"role": "system", "content": "Remember the secret."},
    ]
