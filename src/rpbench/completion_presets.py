"""Versioned named-speaker completion presets for self-play."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .core import load


def resolve_completion_preset(
    config, config_path, player, cards
):
    reference = config.get("completion_preset")
    if reference is None:
        return None

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Invalid completion_preset path")

    if config["backend"].get("mode", "chat") != "completion":
        raise ValueError(
            "completion_preset requires completion mode"
        )

    path = (
        Path(config_path).resolve().parent / reference
    ).resolve()
    preset = load(path)

    if preset.get("style") != "rp_dialogue_v1":
        raise ValueError("Unsupported completion preset style")

    if len(cards) != 1:
        raise ValueError(
            "Named-speaker preset currently requires one NPC card"
        )

    player_name = player["name"]
    npc_name = cards[0][1]["name"]

    if (
        not isinstance(player_name, str)
        or not player_name.strip()
        or not isinstance(npc_name, str)
        or not npc_name.strip()
        or player_name == npc_name
    ):
        raise ValueError("Invalid or ambiguous speaker names")

    fields = (
        "separator",
        "system_prefix",
        "scene_label",
    )
    for field in fields:
        if not isinstance(preset.get(field), str):
            raise ValueError(f"Invalid completion preset field: {field}")

    if type(preset.get("opening_as_scene")) is not bool:
        raise ValueError("Invalid opening_as_scene")

    shared = {
        "style": preset["style"],
        "separator": preset["separator"],
        "system_prefix": preset["system_prefix"],
        "scene_label": preset["scene_label"],
        "opening_as_scene": preset["opening_as_scene"],
    }

    player_format = {
        **shared,
        "speakers": {
            "user": npc_name,
            "assistant": player_name,
        },
    }

    npc_format = {
        **shared,
        "opening_as_scene": False,
        "speakers": {
            "user": player_name,
            "assistant": npc_name,
        },
    }

    provenance = {
        "id": preset["id"],
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content": preset,
    }

    return npc_format, player_format, provenance
