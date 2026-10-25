"""Structural tests for SillyTavern completion presets."""

import copy
import json

import pytest

from rpbench.prompt_st_completion import (
    SillyTavernContextPreset,
    SillyTavernInstructPreset,
    format_st_instruct_message,
)


def context_payload():
    return {
        "name": "Test Context",
        "story_string": "{{system}}\n{{description}}",
        "story_string_position": 0,
        "story_string_depth": 1,
        "story_string_role": 0,
        "chat_start": "***",
        "example_separator": "***",
        "unknown_future_field": {"enabled": True},
    }


def instruct_payload():
    return {
        "name": "Test Instruct",
        "input_sequence": "### Instruction:",
        "output_sequence": "### Response:",
        "input_suffix": "\n\n",
        "output_suffix": "\n\n",
        "wrap": True,
        "macro": True,
        "sequences_as_stop_strings": True,
        "story_string_prefix": "",
        "story_string_suffix": "\n\n",
        "names_behavior": "force",
    }


@pytest.mark.parametrize(
    ("cls", "factory"),
    [
        (SillyTavernContextPreset, context_payload),
        (SillyTavernInstructPreset, instruct_payload),
    ],
)
def test_lossless_roundtrip(cls, factory, tmp_path):
    payload = factory()
    path = tmp_path / "preset.json"
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    preset = cls.from_file(path)
    assert preset.to_dict() == payload

    original = copy.deepcopy(payload)
    payload.clear()
    assert preset.to_dict() == original

    exported = preset.to_dict()
    exported["name"] = "Changed"
    assert preset.to_dict() == original


@pytest.mark.parametrize(
    "field",
    [
        "story_string_position",
        "story_string_depth",
        "story_string_role",
    ],
)
def test_context_rejects_boolean_integer_fields(field):
    payload = context_payload()
    payload[field] = True

    with pytest.raises(ValueError):
        SillyTavernContextPreset(payload)


def test_context_rejects_missing_story_string():
    with pytest.raises(ValueError):
        SillyTavernContextPreset({
            "name": "Incomplete",
        })


def test_instruct_rejects_invalid_sequences():
    payload = instruct_payload()
    payload["output_sequence"] = 123

    with pytest.raises(ValueError):
        SillyTavernInstructPreset(payload)


def test_instruct_rejects_invalid_flags():
    payload = instruct_payload()
    payload["wrap"] = "yes"

    with pytest.raises(ValueError):
        SillyTavernInstructPreset(payload)


from rpbench.prompt_st_completion import render_st_story_string


def test_story_string_variables():
    preset = SillyTavernContextPreset({
        "story_string": "{{system}}\n{{description}}",
    })

    assert render_st_story_string(
        preset,
        {
            "system": "Main instructions",
            "description": "Elara is an archivist.",
        },
    ) == "Main instructions\nElara is an archivist."


def test_story_string_optional_sections():
    preset = SillyTavernContextPreset({
        "story_string": (
            "{{#if system}}{{system}}\n{{/if}}"
            "{{#if scenario}}{{scenario}}\n{{/if}}"
            "{{trim}}"
        ),
    })

    assert render_st_story_string(
        preset,
        {"system": "Main", "scenario": ""},
    ) == "Main"


def test_story_string_rejects_unsupported_tags():
    preset = SillyTavernContextPreset({
        "story_string": "{{#each characters}}{{this}}{{/each}}",
    })

    with pytest.raises(ValueError):
        render_st_story_string(preset, {})


def test_story_string_rejects_nested_conditionals():
    preset = SillyTavernContextPreset({
        "story_string": (
            "{{#if system}}"
            "{{#if scenario}}Nested{{/if}}"
            "{{/if}}"
        ),
    })

    with pytest.raises(ValueError):
        render_st_story_string(
            preset, {"system": "Yes", "scenario": "Yes"}
        )


def test_story_string_rejects_invalid_values():
    preset = SillyTavernContextPreset({
        "story_string": "{{system}}",
    })

    with pytest.raises(ValueError):
        render_st_story_string(preset, {"system": 123})



def test_trim_removes_adjacent_newlines():
    preset = SillyTavernContextPreset({
        "story_string": "First\n{{trim}}\nSecond",
    })

    assert render_st_story_string(
        preset, {}
    ) == "FirstSecond"


def test_trim_preserves_non_newline_whitespace():
    preset = SillyTavernContextPreset({
        "story_string": "First \n{{trim}}\n Second",
    })

    assert render_st_story_string(
        preset, {}
    ) == "First  Second"



def test_instruct_message_user():
    preset = SillyTavernInstructPreset(instruct_payload())

    assert format_st_instruct_message(
        preset, "user", "Hello"
    ) == "### Instruction:\nHello\n\n"


def test_instruct_message_assistant():
    preset = SillyTavernInstructPreset(instruct_payload())

    assert format_st_instruct_message(
        preset, "assistant", "Greetings"
    ) == "### Response:\nGreetings\n\n"


def test_instruct_message_wrap_fallback():
    preset = SillyTavernInstructPreset({
        "input_sequence": "[USER]",
        "output_sequence": "[ASSISTANT]",
        "wrap": True,
    })

    assert format_st_instruct_message(
        preset, "user", "Hello"
    ) == "[USER]\nHello\n"


def test_instruct_message_system_same_as_user():
    preset = SillyTavernInstructPreset({
        "input_sequence": "[USER]",
        "input_suffix": "[END]",
        "system_same_as_user": True,
    })

    assert format_st_instruct_message(
        preset, "system", "Instructions"
    ) == "[USER]Instructions[END]"


def test_instruct_message_named_speaker():
    preset = SillyTavernInstructPreset({
        "input_sequence": "[{{name}}]",
        "names_behavior": "always",
        "macro": True,
        "wrap": True,
    })

    assert format_st_instruct_message(
        preset, "user", "Hello", name="Rowan"
    ) == "[Rowan]\nRowan: Hello\n"



from rpbench.prompt_st_completion import (
    format_st_generation_prefix,
)


def test_generation_prefix_uses_last_output():
    preset = SillyTavernInstructPreset({
        "output_sequence": "### Response:",
        "last_output_sequence": "### Final:",
        "wrap": True,
    })

    assert format_st_generation_prefix(
        preset
    ) == "\n### Final:\n"


def test_generation_prefix_falls_back_to_output():
    preset = SillyTavernInstructPreset({
        "output_sequence": "[ASSISTANT]",
        "wrap": False,
    })

    assert format_st_generation_prefix(
        preset
    ) == "[ASSISTANT]"


def test_generation_prefix_named_speaker():
    preset = SillyTavernInstructPreset({
        "output_sequence": "[{{name}}]",
        "macro": True,
        "names_behavior": "always",
        "wrap": True,
    })

    assert format_st_generation_prefix(
        preset, name="Elara"
    ) == "\n[Elara]\nElara:"


def test_generation_prefix_validates_name():
    preset = SillyTavernInstructPreset({
        "output_sequence": "[AI]",
    })

    with pytest.raises(ValueError):
        format_st_generation_prefix(preset, name=123)


def test_generation_prefix_does_not_mutate_preset():
    preset = SillyTavernInstructPreset({
        "output_sequence": "[AI]",
        "wrap": True,
    })

    before = preset.to_dict()
    format_st_generation_prefix(preset)

    assert preset.to_dict() == before
