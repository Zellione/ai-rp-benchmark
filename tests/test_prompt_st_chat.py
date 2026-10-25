from pathlib import Path

import pytest

from rpbench.prompt_chat import ChatPromptConfig
from rpbench.prompt_models import PromptMessage
from rpbench.prompt_presets import SillyTavernChatPreset
from rpbench.prompt_st_chat import render_st_chat_preset


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "sillytavern"
    / "default-openai.json"
)

MARKERS = {
    "worldInfoBefore": "",
    "charDescription": "Astronomer and observatory keeper.",
    "charPersonality": "Reserved but observant.",
    "scenario": "Rain falls on the coastal observatory.",
    "worldInfoAfter": "",
    "dialogueExamples": "",
    "personaDescription": "Rowan is a returning engineer.",
}


def load_preset():
    return SillyTavernChatPreset.from_file(FIXTURE)


def test_standard_preset_renders_npc_first_chat():
    messages = render_st_chat_preset(
        load_preset(),
        [
            PromptMessage("assistant", "Elara's opening"),
            PromptMessage("user", "Rowan returns"),
        ],
        MARKERS,
        character_name="Elara",
        user_name="Rowan",
        config=ChatPromptConfig(
            user_filler="[Start a new Chat]",
            require_user_first=True,
        ),
    )

    assert messages[0] == {
        "role": "system",
        "content": (
            "Write Elara's next reply in a fictional chat "
            "between Elara and Rowan."
        ),
    }

    assert [m["role"] for m in messages[-3:]] == [
        "user", "assistant", "user",
    ]

    assert messages[-3]["content"] == "[Start a new Chat]"
    assert messages[-2]["content"] == "Elara's opening"
    assert messages[-1]["content"] == "Rowan returns"

    assert all(m["content"].strip() for m in messages)


def test_persona_order_group():
    messages = render_st_chat_preset(
        load_preset(),
        [PromptMessage("user", "Hello")],
        MARKERS,
        character_name="Elara",
        user_name="Rowan",
        character_id=100001,
    )

    contents = [m["content"] for m in messages]

    assert "Rowan is a returning engineer." in contents

    assert contents.index(
        "Rowan is a returning engineer."
    ) < contents.index(
        "Astronomer and observatory keeper."
    )


def test_missing_active_marker_rejected():
    markers = dict(MARKERS)
    del markers["charDescription"]

    with pytest.raises(ValueError, match="charDescription"):
        render_st_chat_preset(
            load_preset(),
            [PromptMessage("user", "Hello")],
            markers,
            character_name="Elara",
            user_name="Rowan",
        )


def test_empty_default_prompts_omitted():
    messages = render_st_chat_preset(
        load_preset(),
        [PromptMessage("user", "Hello")],
        MARKERS,
        character_name="Elara",
        user_name="Rowan",
    )

    assert not any(m["content"] == "" for m in messages)
    assert not any(
        "enhanceDefinitions" in m["content"]
        for m in messages
    )


def test_missing_history_marker_rejected():
    payload = load_preset().to_dict()

    for group in payload["prompt_order"]:
        group["order"] = [
            entry
            for entry in group["order"]
            if entry["identifier"] != "chatHistory"
        ]

    preset = SillyTavernChatPreset(payload)

    with pytest.raises(ValueError, match="chatHistory"):
        render_st_chat_preset(
            preset,
            [PromptMessage("user", "Hello")],
            MARKERS,
            character_name="Elara",
            user_name="Rowan",
        )
