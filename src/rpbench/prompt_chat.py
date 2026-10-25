"""Deterministic chat prompt composition.

This is a foundation for SillyTavern-compatible chat rendering,
not yet a complete reproduction of SillyTavern's Prompt Manager.
"""

from dataclasses import dataclass
from typing import Sequence

from .prompt_models import PromptMessage, PromptSection


@dataclass(frozen=True)
class ChatPromptConfig:
    user_filler: str | None = None
    require_user_first: bool = False

    def __post_init__(self):
        if not isinstance(self.require_user_first, bool):
            raise ValueError("require_user_first must be boolean")

        if self.user_filler is not None:
            if (
                not isinstance(self.user_filler, str)
                or not self.user_filler.strip()
            ):
                raise ValueError("user_filler must be nonempty text")


def render_chat_prompt(
    sections: Sequence[PromptSection],
    history: Sequence[PromptMessage],
    config: ChatPromptConfig | None = None,
) -> list[dict[str, str]]:
    """Compose a chat request without mutating its inputs.

    Sections appear before conversation history, in the supplied
    order. Future SillyTavern-compatible renderers can place
    additional sections within the history.
    """

    cfg = config or ChatPromptConfig()

    messages = [
        PromptMessage(section.role, section.content)
        for section in sections
        if section.enabled
    ]

    messages.extend(history)

    if cfg.user_filler is not None:
        # Insert a filler only when the first conversational
        # message is an assistant greeting.
        first_non_system = next(
            (
                index
                for index, message in enumerate(messages)
                if message.role != "system"
            ),
            None,
        )

        if (
            first_non_system is not None
            and messages[first_non_system].role == "assistant"
        ):
            messages.insert(
                first_non_system,
                PromptMessage("user", cfg.user_filler),
            )

    if cfg.require_user_first:
        conversational = [
            message.role
            for message in messages
            if message.role != "system"
        ]

        if conversational and conversational[0] != "user":
            raise ValueError(
                "Chat history must start with a user message"
            )

        for previous, current in zip(
            conversational,
            conversational[1:],
        ):
            if previous == current:
                raise ValueError(
                    "Conversation roles must alternate"
                )

    return [message.to_dict() for message in messages]


def render_ordered_chat_prompt(
    sections: Sequence[PromptSection],
    history: Sequence[PromptMessage],
    order: Sequence[str],
    config: ChatPromptConfig | None = None,
) -> list[dict[str, str]]:
    """Render sections around an explicit chat_history anchor.

    A section may appear only once. Disabled sections produce no
    message. Post-history sections retain their declared role.

    This is a deterministic Prompt Manager foundation, not a full
    SillyTavern implementation.
    """
    if (
        not isinstance(order, (list, tuple))
        or any(not isinstance(value, str) for value in order)
    ):
        raise ValueError("Prompt order must be a sequence of IDs")

    section_map = {}

    for section in sections:
        if section.id == "chat_history":
            raise ValueError("Reserved prompt section ID")

        if section.id in section_map:
            raise ValueError(
                f"Duplicate prompt section: {section.id}"
            )

        section_map[section.id] = section

    if list(order).count("chat_history") != 1:
        raise ValueError(
            "Prompt order requires exactly one chat_history anchor"
        )

    if len(set(order)) != len(order):
        raise ValueError("Duplicate prompt order entry")

    unknown = set(order) - set(section_map) - {"chat_history"}

    if unknown:
        raise ValueError(
            f"Unknown prompt sections: {sorted(unknown)}"
        )

    missing = set(section_map) - set(order)

    if missing:
        raise ValueError(
            f"Unplaced prompt sections: {sorted(missing)}"
        )

    messages = []

    for identifier in order:
        if identifier == "chat_history":
            messages.extend(history)
        else:
            section = section_map[identifier]

            if section.enabled:
                messages.append(
                    PromptMessage(section.role, section.content)
                )

    # Apply the same filler and alternation handling as the
    # original renderer, without duplicating those rules.
    return render_chat_prompt([], messages, config)


def validate_strict_chat_template(
    messages: Sequence[dict[str, str]],
) -> None:
    """Validate a system-prefix, user-first chat template.

    Intended for templates with Mistral-like role restrictions.
    This validator is opt-in and does not modify messages.
    """
    previous_role = None
    conversation_started = False

    for index, message in enumerate(messages):
        role = message["role"]

        if role == "system":
            if conversation_started:
                raise ValueError(
                    f"System message after conversation start: {index}"
                )
            continue

        conversation_started = True

        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Unsupported conversation role: {role}"
            )

        if previous_role is None and role != "user":
            raise ValueError(
                "Conversation must begin with a user message"
            )

        if role == previous_role:
            raise ValueError(
                f"Consecutive {role} messages at index {index}"
            )

        previous_role = role
