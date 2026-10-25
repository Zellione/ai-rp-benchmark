"""Integrated ST completion prompt and backend tests."""

from copy import deepcopy

import pytest

from rpbench.core import Backend
from rpbench.prompt_st_completion import (
    render_st_completion_prompt,
)


def make_format():
    return {
        "style": "st_instruct_v1",
        "user_name": "Rowan",
        "character_name": "Elara",
        "context_preset": {
            "story_string": "{{system}}\n{{trim}}",
            "story_string_position": 0,
            "chat_start": "",
        },
        "instruct_preset": {
            "input_sequence": "[USER]",
            "output_sequence": "[AI]",
            "input_suffix": "[END]",
            "output_suffix": "[END]",
            "wrap": False,
            "macro": False,
            "names_behavior": "force",
        },
    }


def make_messages():
    return [
        {"role": "system", "content": "NPC instructions"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Greetings"},
        {"role": "user", "content": "What happened?"},
    ]


def test_complete_prompt():
    result = render_st_completion_prompt(
        make_messages(), make_format()
    )

    assert result == (
        "NPC instructions"
        "[USER]Hello[END]"
        "[AI]Greetings[END]"
        "[USER]What happened?[END]"
        "[AI]"
    )


def test_backend_integration():
    backend = Backend(
        name="llamacpp",
        base_url="http://127.0.0.1:8080",
        model="test",
        mode="completion",
    )

    url, payload = backend.prepare(
        make_messages(),
        {"max_tokens": 128},
        make_format(),
    )

    assert url == "http://127.0.0.1:8080/v1/completions"
    assert payload["prompt"] == render_st_completion_prompt(
        make_messages(), make_format()
    )
    assert payload["max_tokens"] == 128


def test_original_messages_and_format_unchanged():
    messages = make_messages()
    fmt = make_format()
    before_messages = deepcopy(messages)
    before_format = deepcopy(fmt)

    render_st_completion_prompt(messages, fmt)

    assert messages == before_messages
    assert fmt == before_format


def test_in_chat_story_position_rejected():
    fmt = make_format()
    fmt["context_preset"]["story_string_position"] = 1

    with pytest.raises(ValueError):
        render_st_completion_prompt(make_messages(), fmt)


def test_context_cannot_override_system():
    fmt = make_format()
    fmt["story_values"] = {"system": "Different context"}

    with pytest.raises(ValueError):
        render_st_completion_prompt(make_messages(), fmt)


def test_legacy_completion_style_unchanged():
    backend = Backend(
        name="llamacpp",
        base_url="http://127.0.0.1:8080",
        model="test",
        mode="completion",
    )

    _, payload = backend.prepare(
        make_messages(),
        {},
        {"style": "rp_dialogue_v1",
         "speakers": {"user": "Rowan", "assistant": "Elara"}},
    )

    assert payload["prompt"].endswith("Elara:\n")
