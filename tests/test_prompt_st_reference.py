"""Reference tests for SillyTavern in-chat injection ordering.

Algorithm reference:
SillyTavern/public/scripts/openai.js
populationInjectionPrompts()

Scope:
- Text-only prompt injections
- No extension prompts
- No token-budget trimming
- No downstream provider transformations

The reference deliberately uses reverse-chronological
construction rather than RPBench's chronological buckets.
"""

from itertools import product

import pytest

from rpbench.prompt_injections import (
    ChatInjection,
    apply_chat_injections,
)
from rpbench.prompt_models import PromptMessage


ROLES = ("system", "user", "assistant")


def reference_injections(history, injections):
    """Model SillyTavern's reverse-history splice algorithm."""
    messages = [
        (message.role, message.content)
        for message in reversed(history)
    ]

    inserted = 0
    max_depth = max(
        (injection.depth for injection in injections),
        default=0,
    )

    for depth in range(max_depth + 1):
        selected = [
            injection
            for injection in injections
            if injection.depth == depth
            and injection.content
        ]

        groups = {}

        for injection in selected:
            groups.setdefault(injection.order, []).append(
                injection
            )

        role_messages = []

        for order in sorted(groups, reverse=True):
            for role in ROLES:
                content = "\n".join(
                    injection.content
                    for injection in groups[order]
                    if injection.role == role
                )

                if content:
                    role_messages.append((role, content))

        if role_messages:
            index = min(depth + inserted, len(messages))
            messages[index:index] = role_messages
            inserted += len(role_messages)

    return list(reversed(messages))


def render_pairs(history, injections):
    return [
        (message.role, message.content)
        for message in apply_chat_injections(
            history,
            injections,
        )
    ]


@pytest.mark.parametrize("history_size", [0, 1, 2, 5])
@pytest.mark.parametrize("depth", [0, 1, 2, 5, 10])
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("order", [-10, 0, 100, 200])
def test_single_injection_reference(
    history_size,
    depth,
    role,
    order,
):
    history = [
        PromptMessage(
            "user" if index % 2 == 0 else "assistant",
            f"History {index}",
        )
        for index in range(history_size)
    ]

    injections = [
        ChatInjection(
            "test",
            role,
            "Injected",
            depth,
            order,
        )
    ]

    assert render_pairs(
        history, injections
    ) == reference_injections(
        history, injections
    )


@pytest.mark.parametrize(
    ("orders", "roles"),
    list(product(
        [
            (100, 100, 100),
            (200, 100, 10),
            (10, 200, 100),
            (0, -10, 200),
        ],
        [
            ("system", "user", "assistant"),
            ("assistant", "user", "system"),
            ("system", "system", "system"),
            ("user", "assistant", "user"),
        ],
    )),
)
def test_grouping_reference(orders, roles):
    history = [
        PromptMessage("user", "Opening"),
        PromptMessage("assistant", "Response"),
        PromptMessage("user", "Follow-up"),
    ]

    injections = [
        ChatInjection(
            f"injection-{index}",
            role,
            f"Content {index}",
            1,
            order,
        )
        for index, (role, order) in enumerate(
            zip(roles, orders)
        )
    ]

    assert render_pairs(
        history, injections
    ) == reference_injections(
        history, injections
    )


def test_multiple_depths_reference():
    history = [
        PromptMessage("user", "First"),
        PromptMessage("assistant", "Second"),
        PromptMessage("user", "Third"),
    ]

    injections = [
        ChatInjection("a", "system", "A", 0, 100),
        ChatInjection("b", "assistant", "B", 1, 200),
        ChatInjection("c", "user", "C", 2, 10),
        ChatInjection("d", "system", "D", 3, 100),
        ChatInjection("e", "assistant", "E", 8, 100),
    ]

    assert render_pairs(
        history, injections
    ) == reference_injections(
        history, injections
    )
