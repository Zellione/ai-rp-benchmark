"""Known differences between RPBench and SillyTavern.

Upstream reference:
SillyTavern/public/scripts/openai.js
populationInjectionPrompts()

These tests document compatibility requirements.
"""

from rpbench.prompt_injections import (
    ChatInjection,
    apply_chat_injections,
)
from rpbench.prompt_models import PromptMessage


def injection(identifier, content, order):
    return ChatInjection(
        identifier=identifier,
        role="system",
        content=content,
        depth=1,
        order=order,
    )


def test_different_orders_remain_separate_messages():
    history = [
        PromptMessage("user", "First"),
        PromptMessage("assistant", "Second"),
    ]

    result = apply_chat_injections(
        history,
        [
            injection("low", "Low order", 10),
            injection("high", "High order", 200),
        ],
    )

    injected = [
        message for message in result
        if message.content in {"Low order", "High order"}
    ]

    assert len(injected) == 2


def test_same_role_and_order_are_grouped():
    history = [
        PromptMessage("user", "First"),
        PromptMessage("assistant", "Second"),
    ]

    result = apply_chat_injections(
        history,
        [
            injection("one", "First injection", 100),
            injection("two", "Second injection", 100),
        ],
    )

    assert sum(
        message.content == "First injection\nSecond injection"
        for message in result
    ) == 1
