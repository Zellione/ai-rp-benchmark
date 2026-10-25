"""Portable configuration for opt-in SillyTavern NPC rendering."""

from pathlib import Path

from .prompt_presets import SillyTavernChatPreset
from .simulation_prompts import render_npc_chat_messages


def load_st_npc_config(runtime, config_path, npc_system):
    """Validate and snapshot an optional NPC chat preset."""
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")

    settings = runtime.get("st_chat_npc")
    if settings is None:
        return None

    if not isinstance(settings, dict):
        raise ValueError("st_chat_npc must be an object")

    required = {
        "preset", "character_name", "user_name",
        "context_marker",
    }
    allowed = required | {"squash_system_messages"}

    if set(settings) - allowed:
        raise ValueError("Unknown st_chat_npc settings")

    for key in required:
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid st_chat_npc.{key}")

    squash = settings.get("squash_system_messages", False)
    if not isinstance(squash, bool):
        raise ValueError("Invalid squash_system_messages")

    path = (
        Path(config_path).resolve().parent / settings["preset"]
    ).resolve()

    preset = SillyTavernChatPreset.from_file(path)
    marker = settings["context_marker"]

    ordered = preset.ordered_prompts()
    matches = [
        item for item in ordered
        if item["identifier"] == marker
        and item["enabled"]
        and item.get("marker", False)
        and item.get("injection_position", 0) == 0
    ]

    if len(matches) != 1 or marker == "chatHistory":
        raise ValueError(
            "Context marker must be an enabled relative "
            "marker prompt distinct from chatHistory"
        )

    if not isinstance(npc_system, str) or not npc_system.strip():
        raise ValueError("NPC system context must be nonempty")

    # Validate the entire currently supported prompt subset
    # before a simulation run is created.
    render_npc_chat_messages(
        [{"role": "system", "content": npc_system}],
        preset,
        {marker: npc_system},
        character_name=settings["character_name"],
        user_name=settings["user_name"],
        squash_system_messages=squash,
    )

    return {
        "mode": "sillytavern_chat",
        "preset_path": str(path),
        "preset": preset.to_dict(),
        "character_name": settings["character_name"],
        "user_name": settings["user_name"],
        "context_marker": marker,
        "squash_system_messages": squash,
    }


def render_st_npc_request(messages, archived):
    """Render using only the archived configuration and messages."""
    if archived is None:
        return [dict(message) for message in messages]

    if not isinstance(archived, dict):
        raise ValueError("Invalid archived NPC renderer")

    if archived.get("mode") != "sillytavern_chat":
        raise ValueError("Unsupported archived NPC renderer")

    if not messages or messages[0].get("role") != "system":
        raise ValueError("Missing NPC system context")

    marker = archived["context_marker"]
    preset = SillyTavernChatPreset(archived["preset"])

    return render_npc_chat_messages(
        messages,
        preset,
        {marker: messages[0]["content"]},
        character_name=archived["character_name"],
        user_name=archived["user_name"],
        squash_system_messages=archived["squash_system_messages"],
    )
