import pytest

from rpbench.prompt_chat import (
    ChatPromptConfig,
    render_chat_prompt,
)
from rpbench.prompt_models import (
    PromptMessage,
    PromptSection,
)


def test_chat_prompt_order():
    sections = [
        PromptSection("main", "system", "Main prompt"),
        PromptSection("character", "system", "Character card"),
        PromptSection("scenario", "system", "Scenario"),
    ]
    history = [
        PromptMessage("user", "Hello"),
        PromptMessage("assistant", "Hello back"),
    ]

    messages = render_chat_prompt(sections, history)

    assert [message["content"] for message in messages] == [
        "Main prompt",
        "Character card",
        "Scenario",
        "Hello",
        "Hello back",
    ]


def test_disabled_section():
    sections = [
        PromptSection("main", "system", "Main"),
        PromptSection(
            "disabled", "system", "", enabled=False
        ),
    ]

    messages = render_chat_prompt(sections, [])

    assert messages == [
        {"role": "system", "content": "Main"},
    ]


def test_npc_first_user_filler():
    sections = [
        PromptSection("main", "system", "Play Elara"),
    ]
    history = [
        PromptMessage("assistant", "Elara's greeting"),
        PromptMessage("user", "Rowan's first turn"),
    ]

    messages = render_chat_prompt(
        sections,
        history,
        ChatPromptConfig(
            user_filler="[Start a new chat]",
            require_user_first=True,
        ),
    )

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert messages[1]["content"] == "[Start a new chat]"


def test_no_duplicate_filler():
    messages = render_chat_prompt(
        [],
        [
            PromptMessage("user", "Already starts with user"),
            PromptMessage("assistant", "Greeting"),
        ],
        ChatPromptConfig(
            user_filler="Filler",
            require_user_first=True,
        ),
    )

    assert len(messages) == 2


def test_invalid_alternation():
    with pytest.raises(ValueError):
        render_chat_prompt(
            [],
            [
                PromptMessage("user", "First"),
                PromptMessage("user", "Second"),
            ],
            ChatPromptConfig(require_user_first=True),
        )


def test_inputs_unchanged():
    sections = [
        PromptSection("main", "system", "Main")
    ]
    history = [
        PromptMessage("assistant", "Greeting")
    ]

    render_chat_prompt(
        sections,
        history,
        ChatPromptConfig(user_filler="Filler"),
    )

    assert len(sections) == 1
    assert len(history) == 1
    assert history[0].role == "assistant"


def test_invalid_filler():
    with pytest.raises(ValueError):
        ChatPromptConfig(user_filler="   ")
