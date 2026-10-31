"""Portable configuration for opt-in SillyTavern NPC rendering."""

import hashlib
from pathlib import Path
from .simulation_megumin import (
    load_megumin_npc, render_megumin_npc,
)
from .simulation_arfyjb import (
    load_arfyjb_npc, render_arfyjb_npc,
)

from .prompt_presets import SillyTavernChatPreset
from .simulation_prompts import render_npc_chat_messages
from .preset_backend_compat import (
    lower_to_alternating_chat,
    lower_to_alternating_chat_v2,
)
from .prompt_chat import validate_strict_chat_template


def load_st_npc_config(runtime, config_path, npc_system):
    """Validate and snapshot an optional NPC chat preset."""
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")

    active = [
        key for key in ("st_chat_npc", "arfyjb_npc", "megumin_npc")
        if key in runtime
    ]
    if len(active) > 1:
        raise ValueError("Multiple NPC preset renderers")

    if "megumin_npc" in runtime:
        return load_megumin_npc(runtime, config_path, npc_system)

    if "arfyjb_npc" in runtime:
        if "st_chat_npc" in runtime:
            raise ValueError("Multiple NPC preset renderers")
        return load_arfyjb_npc(runtime, config_path)

    settings = runtime.get("st_chat_npc")
    if settings is None:
        return None

    if not isinstance(settings, dict):
        raise ValueError("st_chat_npc must be an object")

    required = {
        "preset", "character_name", "user_name",
        "context_marker",
    }
    allowed = required | {
        "squash_system_messages", "character_id", "backend_compat"
    }

    if set(settings) - allowed:
        raise ValueError("Unknown st_chat_npc settings")

    for key in required:
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid st_chat_npc.{key}")

    squash = settings.get("squash_system_messages", False)
    if not isinstance(squash, bool):
        raise ValueError("Invalid squash_system_messages")

    character_id = settings.get("character_id", 100000)
    if type(character_id) is not int or character_id < 0:
        raise ValueError("Invalid st_chat_npc.character_id")

    compatibility = settings.get("backend_compat")
    if compatibility not in {
        None, "alternating-chat-v1", "alternating-chat-v2"
    }:
        raise ValueError("Unsupported st_chat_npc.backend_compat")

    path = (
        Path(config_path).resolve().parent / settings["preset"]
    ).resolve()

    preset = SillyTavernChatPreset.from_file(
        path, allow_disabled_orphans=True
    )
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if "squash_system_messages" not in settings:
        squash = preset.to_dict().get(
            "squash_system_messages", False
        )
        if type(squash) is not bool:
            raise ValueError("Invalid preset squash setting")
    marker = settings["context_marker"]

    ordered = preset.ordered_prompts(character_id)
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
    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"] and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }
    markers[marker] = npc_system

    render_npc_chat_messages(
        [{"role": "system", "content": npc_system}],
        preset,
        markers,
        character_name=settings["character_name"],
        user_name=settings["user_name"],
        squash_system_messages=squash,
        character_id=character_id,
        source_sha256=source_sha256,
    )

    return {
        "mode": "sillytavern_chat",
        "preset_path": str(path),
        "source_sha256": source_sha256,
        "preset": preset.to_dict(),
        "character_name": settings["character_name"],
        "user_name": settings["user_name"],
        "context_marker": marker,
        "character_id": character_id,
        "backend_compat": compatibility,
        "squash_system_messages": squash,
    }


def render_st_npc_request(messages, archived):
    """Render using only the archived configuration and messages."""
    if archived is None:
        return [dict(message) for message in messages]

    if not isinstance(archived, dict):
        raise ValueError("Invalid archived NPC renderer")

    if archived.get("mode") == "megumin_v9":
        return render_megumin_npc(messages, archived)

    if archived.get("mode") == "arfyjb_universal":
        return render_arfyjb_npc(messages, archived)

    if archived.get("mode") != "sillytavern_chat":
        raise ValueError("Unsupported archived NPC renderer")

    if not messages or messages[0].get("role") != "system":
        raise ValueError("Missing NPC system context")

    marker = archived["context_marker"]
    preset = SillyTavernChatPreset(
        archived["preset"], allow_disabled_orphans=True
    )
    character_id = archived.get("character_id", 100000)
    ordered = preset.ordered_prompts(character_id)
    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"] and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }
    markers[marker] = messages[0]["content"]

    rendered = render_npc_chat_messages(
        messages,
        preset,
        markers,
        character_name=archived["character_name"],
        user_name=archived["user_name"],
        squash_system_messages=archived["squash_system_messages"],
        character_id=character_id,
        source_sha256=archived.get("source_sha256"),
    )

    compatibility = archived.get("backend_compat")
    if compatibility is None:
        return rendered

    strategies = {
        "alternating-chat-v1": lower_to_alternating_chat,
        "alternating-chat-v2": lower_to_alternating_chat_v2,
    }
    if compatibility not in strategies:
        raise ValueError("Unsupported archived backend compatibility")

    lowered, _metadata = strategies[compatibility](rendered)
    validate_strict_chat_template(lowered)
    return lowered
