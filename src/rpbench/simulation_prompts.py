"""Optional NPC prompt rendering for RPBench simulations.

This module does not change the legacy simulation pipeline.
It provides a reusable adapter for future simulation/resume
integration.
"""

from collections.abc import Mapping, Sequence

from .prompt_models import PromptMessage
from .prompt_presets import SillyTavernChatPreset
from .prompt_st_chat import render_st_chat_preset
from .preset_adapter import (
    SillyTavernPresetAdapter, render_with_adapter,
)


def render_npc_chat_messages(
    messages: Sequence[dict[str, str]],
    preset: SillyTavernChatPreset,
    marker_content: Mapping[str, str],
    *,
    character_name: str,
    user_name: str,
    squash_system_messages: bool = False,
    character_id: int = 100000,
    source_sha256: str | None = None,
) -> list[dict[str, str]]:
    """Render an NPC request with a SillyTavern chat preset.

    The caller supplies an existing legacy NPC message history.
    Its first system message is the legacy system prompt.

    The preset owns the new system-prompt construction.
    The first legacy system message is therefore excluded from
    the conversational history to avoid duplication.
    """
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a sequence")

    if not messages:
        raise ValueError("NPC messages must not be empty")

    first = messages[0]

    if (
        not isinstance(first, dict)
        or first.get("role") != "system"
        or not isinstance(first.get("content"), str)
        or not first["content"].strip()
    ):
        raise ValueError(
            "NPC history must begin with a system message"
        )

    history = []

    for message in messages[1:]:
        if not isinstance(message, dict):
            raise ValueError("Invalid NPC message")

        if set(message) != {"role", "content"}:
            raise ValueError(
                "Only text-only NPC messages are supported"
            )

        history.append(
            PromptMessage(
                message["role"],
                message["content"],
            )
        )

    if source_sha256 is not None:
        adapter = SillyTavernPresetAdapter.from_snapshot(
            preset.to_dict(),
            source_sha256,
            squash_system_messages=squash_system_messages,
        )
        return render_with_adapter(
            adapter,
            history,
            markers=marker_content,
            character_name=character_name,
            user_name=user_name,
            character_id=character_id,
        ).messages

    # Legacy snapshots retain their historical rendering behavior.
    return render_st_chat_preset(
        preset,
        history,
        marker_content,
        character_name=character_name,
        user_name=user_name,
        character_id=character_id,
        squash_system_messages=squash_system_messages,
    )
