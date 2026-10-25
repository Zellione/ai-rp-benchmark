"""Explicit adapter from SillyTavern chat presets to RPBench messages.

This is a structural subset, not a complete SillyTavern renderer.
"""

from collections.abc import Mapping, Sequence

from .prompt_chat import (
    ChatPromptConfig,
    render_chat_prompt,
)
from .prompt_models import PromptMessage
from .prompt_presets import (
    DEFAULT_CHARACTER_ID,
    SillyTavernChatPreset,
)
from .prompt_injections import ChatInjection, apply_chat_injections


def _expand_names(
    content: str,
    character_name: str,
    user_name: str,
) -> str:
    return (
        content.replace("{{char}}", character_name)
        .replace("{{user}}", user_name)
    )


def _squash_plain_system_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge adjacent plain system messages with newline separators.

    Supported subset of SillyTavern ChatCompletion.squashSystemMessages.
    Internal control-message identifiers and named messages are not
    represented by RPBench's current text-only message type.
    """
    result: list[dict[str, str]] = []

    for message in messages:
        current = message.copy()

        if (
            result
            and current["role"] == "system"
            and result[-1]["role"] == "system"
        ):
            result[-1]["content"] += "\n" + current["content"]
        else:
            result.append(current)

    return result


def render_st_chat_preset(
    preset: SillyTavernChatPreset,
    history: Sequence[PromptMessage],
    marker_content: Mapping[str, str],
    *,
    character_name: str,
    user_name: str,
    character_id: int = DEFAULT_CHARACTER_ID,
    config: ChatPromptConfig | None = None,
    squash_system_messages: bool = False,
) -> list[dict[str, str]]:
    """Render the supported SillyTavern chat-preset subset.

    Relative prompts follow Prompt Manager ordering. In-chat
    prompts are injected into the original history at their
    configured depth, independent of manager ordering.
    """
    if not isinstance(character_name, str) or not character_name.strip():
        raise ValueError("character_name must be nonempty")

    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("user_name must be nonempty")

    if not isinstance(marker_content, Mapping):
        raise ValueError("marker_content must be a mapping")

    if not isinstance(squash_system_messages, bool):
        raise ValueError("squash_system_messages must be boolean")

    ordered = preset.ordered_prompts(character_id)

    injections: list[ChatInjection] = []
    relative: list[tuple[str, PromptMessage | None]] = []
    history_count = 0

    for item in ordered:
        if not item["enabled"]:
            continue

        identifier = item["identifier"]

        if identifier == "chatHistory":
            history_count += 1
            relative.append(("history", None))
            continue

        if item.get("marker", False):
            if identifier not in marker_content:
                raise ValueError(
                    f"Missing active SillyTavern marker: {identifier}"
                )
            content = marker_content[identifier]
        else:
            content = item.get("content", "")

        if not isinstance(content, str):
            raise ValueError(
                f"Prompt content must be text: {identifier}"
            )

        if not content.strip():
            continue

        content = _expand_names(
            content,
            character_name,
            user_name,
        )

        role = item.get("role", "system")
        position = item.get("injection_position", 0)

        if type(position) is not int or position not in (0, 1):
            raise ValueError(
                f"Unsupported injection position: {identifier}"
            )

        if position == 1:
            if item.get("marker", False):
                raise ValueError(
                    f"In-chat marker not supported: {identifier}"
                )

            injections.append(
                ChatInjection(
                    identifier=identifier,
                    role=role,
                    content=content,
                    depth=item.get("injection_depth", 0),
                    order=item.get("injection_order", 100),
                )
            )
        else:
            relative.append(
                ("message", PromptMessage(role, content))
            )

    if history_count != 1:
        raise ValueError(
            "Exactly one active chatHistory marker required"
        )

    injected_history = apply_chat_injections(
        history,
        injections,
    )

    messages: list[PromptMessage] = []

    for kind, message in relative:
        if kind == "history":
            messages.extend(injected_history)
        else:
            messages.append(message)

    rendered = render_chat_prompt([], messages, config)

    if squash_system_messages:
        return _squash_plain_system_messages(rendered)

    return rendered
