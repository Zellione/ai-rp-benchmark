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


def _expand_names(
    content: str,
    character_name: str,
    user_name: str,
) -> str:
    return (
        content.replace("{{char}}", character_name)
        .replace("{{user}}", user_name)
    )


def render_st_chat_preset(
    preset: SillyTavernChatPreset,
    history: Sequence[PromptMessage],
    marker_content: Mapping[str, str],
    *,
    character_name: str,
    user_name: str,
    character_id: int = DEFAULT_CHARACTER_ID,
    config: ChatPromptConfig | None = None,
) -> list[dict[str, str]]:
    """Render a supported subset of a SillyTavern chat preset.

    Dynamic markers must be supplied explicitly. Empty marker values
    are allowed and omitted. In-chat injections are not supported.
    """
    if not isinstance(character_name, str) or not character_name.strip():
        raise ValueError("character_name must be nonempty")

    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("user_name must be nonempty")

    if not isinstance(marker_content, Mapping):
        raise ValueError("marker_content must be a mapping")

    ordered = preset.ordered_prompts(character_id)

    messages: list[PromptMessage] = []
    history_count = 0

    for item in ordered:
        if not item["enabled"]:
            continue

        identifier = item["identifier"]

        if identifier == "chatHistory":
            history_count += 1
            messages.extend(history)
            continue

        if item.get("marker", False):
            if identifier not in marker_content:
                raise ValueError(
                    f"Missing active SillyTavern marker: {identifier}"
                )

            content = marker_content[identifier]
        else:
            content = item.get("content", "")

            if item.get("position") == 1:
                raise ValueError(
                    f"In-chat injection not supported: {identifier}"
                )

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

        # Official marker prompts generally omit role and default
        # to system. Other roles must be explicitly declared.
        role = item.get("role", "system")

        messages.append(PromptMessage(role, content))

    if history_count != 1:
        raise ValueError(
            "Exactly one active chatHistory marker required"
        )

    return render_chat_prompt([], messages, config)
