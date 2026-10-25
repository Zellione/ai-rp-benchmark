"""Tests for optional SillyTavern-style system squashing."""

import pytest

from rpbench.prompt_models import PromptMessage
from rpbench.prompt_presets import SillyTavernChatPreset
from rpbench.prompt_st_chat import render_st_chat_preset


def preset():
    return SillyTavernChatPreset({
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Main instructions",
            },
            {
                "identifier": "scenario",
                "role": "system",
                "content": "Scenario",
            },
            {
                "identifier": "chatHistory",
                "role": "system",
                "marker": True,
            },
            {
                "identifier": "post",
                "role": "system",
                "content": "Post-history",
            },
        ],
        "prompt_order": [{
            "character_id": 100000,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "scenario", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
                {"identifier": "post", "enabled": True},
            ],
        }],
    })


def render(history, squash=False):
    return render_st_chat_preset(
        preset(),
        history,
        {},
        character_name="Elara",
        user_name="Rowan",
        squash_system_messages=squash,
    )


def test_default_preserves_separate_system_messages():
    result = render([PromptMessage("user", "Hello")])

    assert [m["content"] for m in result] == [
        "Main instructions",
        "Scenario",
        "Hello",
        "Post-history",
    ]


def test_opt_in_merges_adjacent_system_messages():
    result = render(
        [PromptMessage("user", "Hello")],
        squash=True,
    )

    assert result == [
        {
            "role": "system",
            "content": "Main instructions\nScenario",
        },
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Post-history"},
    ]


def test_history_messages_form_boundaries():
    result = render(
        [
            PromptMessage("user", "First"),
            PromptMessage("assistant", "Second"),
            PromptMessage("user", "Third"),
        ],
        squash=True,
    )

    assert [m["role"] for m in result] == [
        "system", "user", "assistant", "user", "system"
    ]


def test_empty_history_merges_all_system_messages():
    result = render([], squash=True)

    assert result == [{
        "role": "system",
        "content": "Main instructions\nScenario\nPost-history",
    }]


def test_squash_does_not_mutate_history():
    history = [PromptMessage("user", "Hello")]
    before = tuple(history)

    render(history, squash=True)

    assert tuple(history) == before


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes", []])
def test_invalid_squash_option_rejected(invalid):
    with pytest.raises(ValueError):
        render_st_chat_preset(
            preset(),
            [],
            {},
            character_name="Elara",
            user_name="Rowan",
            squash_system_messages=invalid,
        )
