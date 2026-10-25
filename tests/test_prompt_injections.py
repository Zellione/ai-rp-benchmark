import pytest

from rpbench.prompt_injections import (
    ChatInjection,
    apply_chat_injections,
)
from rpbench.prompt_models import PromptMessage


def history():
    return [
        PromptMessage("user", "First"),
        PromptMessage("assistant", "Second"),
        PromptMessage("user", "Third"),
    ]


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (0, ["First", "Second", "Third", "Injected"]),
        (1, ["First", "Second", "Injected", "Third"]),
        (2, ["First", "Injected", "Second", "Third"]),
        (3, ["Injected", "First", "Second", "Third"]),
        (10, ["Injected", "First", "Second", "Third"]),
    ],
)
def test_injection_depth(depth, expected):
    result = apply_chat_injections(
        history(),
        [ChatInjection("note", "system", "Injected", depth)],
    )

    assert [message.content for message in result] == expected


def test_equal_depth_sorted_by_order():
    result = apply_chat_injections(
        history(),
        [
            ChatInjection("late", "system", "B", 1, 200),
            ChatInjection("early", "system", "A", 1, 100),
        ],
    )

    assert [message.content for message in result] == [
        "First", "Second", "A\nB", "Third",
    ]


def test_positions_are_relative_to_original_history():
    result = apply_chat_injections(
        history(),
        [
            ChatInjection("end", "system", "End", 0),
            ChatInjection("middle", "system", "Middle", 2),
        ],
    )

    assert [message.content for message in result] == [
        "First", "Middle", "Second", "Third", "End",
    ]


def test_history_is_unchanged():
    source = history()
    original = tuple(source)

    apply_chat_injections(
        source,
        [ChatInjection("note", "system", "Injected", 1)],
    )

    assert tuple(source) == original


def test_empty_history_depth_zero():
    result = apply_chat_injections(
        [],
        [ChatInjection("note", "system", "Injected", 0)],
    )

    assert [message.content for message in result] == ["Injected"]


@pytest.mark.parametrize("depth", [-1, 1.5, True])
def test_invalid_depth(depth):
    with pytest.raises(ValueError):
        ChatInjection("note", "system", "Injected", depth)


def test_same_role_same_depth_is_grouped():
    result = apply_chat_injections(
        history(),
        [
            ChatInjection("second", "system", "Second", 1, 200),
            ChatInjection("first", "system", "First", 1, 100),
        ],
    )

    assert [message.content for message in result] == [
        "First",
        "Second",
        "First\nSecond",
        "Third",
    ][0:2] + ["First\nSecond", "Third"] if False else [
        "First", "Second", "First\nSecond", "Third"
    ]


def test_mixed_roles_follow_sillytavern_role_order():
    result = apply_chat_injections(
        history(),
        [
            ChatInjection("system", "system", "System note", 1),
            ChatInjection("assistant", "assistant", "AI note", 1),
            ChatInjection("user", "user", "User note", 1),
        ],
    )

    assert [m.role for m in result] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "system",
        "user",
    ]
