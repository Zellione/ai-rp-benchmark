import pytest

from rpbench.prompt_chat import (
    ChatPromptConfig,
    render_ordered_chat_prompt,
)
from rpbench.prompt_models import (
    PromptMessage,
    PromptSection,
)


def fixtures():
    sections = [
        PromptSection("main", "system", "Main instructions"),
        PromptSection("character", "system", "Elara card"),
        PromptSection("post", "system", "Post-history rules"),
    ]

    history = [
        PromptMessage("user", "Hello Elara"),
        PromptMessage("assistant", "Hello Rowan"),
    ]

    return sections, history


def test_ordered_history_and_post_history():
    sections, history = fixtures()

    messages = render_ordered_chat_prompt(
        sections,
        history,
        ["main", "character", "chat_history", "post"],
    )

    assert [m["content"] for m in messages] == [
        "Main instructions",
        "Elara card",
        "Hello Elara",
        "Hello Rowan",
        "Post-history rules",
    ]

    assert [m["role"] for m in messages] == [
        "system", "system", "user", "assistant", "system"
    ]


def test_reordered_sections():
    sections, history = fixtures()

    messages = render_ordered_chat_prompt(
        sections,
        history,
        ["post", "chat_history", "character", "main"],
    )

    assert [m["content"] for m in messages] == [
        "Post-history rules",
        "Hello Elara",
        "Hello Rowan",
        "Elara card",
        "Main instructions",
    ]


def test_npc_first_filler():
    sections = [
        PromptSection("main", "system", "NPC instructions"),
    ]
    history = [
        PromptMessage("assistant", "Elara greets Rowan"),
        PromptMessage("user", "Rowan answers"),
    ]

    messages = render_ordered_chat_prompt(
        sections,
        history,
        ["main", "chat_history"],
        ChatPromptConfig(
            user_filler="[Start a new chat]",
            require_user_first=True,
        ),
    )

    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user",
    ]


@pytest.mark.parametrize(
    "order",
    [
        ["main", "character", "post"],
        ["main", "chat_history", "chat_history", "post"],
        ["main", "chat_history", "unknown", "post"],
        ["main", "chat_history", "post"],
        ["main", "character", "chat_history", "post", "post"],
    ],
)
def test_invalid_order(order):
    sections, history = fixtures()

    with pytest.raises(ValueError):
        render_ordered_chat_prompt(sections, history, order)


def test_inputs_unchanged():
    sections, history = fixtures()
    original_sections = tuple(sections)
    original_history = tuple(history)

    render_ordered_chat_prompt(
        sections,
        history,
        ["main", "character", "chat_history", "post"],
    )

    assert tuple(sections) == original_sections
    assert tuple(history) == original_history
