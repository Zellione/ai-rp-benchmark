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

    if preset.get("style") == "st_instruct_v1":
        from .prompt_st_completion import (
            render_st_completion_prompt,
        )

        if len(cards) != 1:
            raise ValueError(
                "ST instruct preset requires exactly one NPC card"
            )

        if not isinstance(preset.get("id"), str) or not preset["id"].strip():
            raise ValueError("ST instruct preset requires an ID")

        player_name = player["name"]
        npc_name = cards[0][1]["name"]

        if (
            not isinstance(player_name, str)
            or not player_name.strip()
            or not isinstance(npc_name, str)
            or not npc_name.strip()
            or player_name == npc_name
        ):
            raise ValueError("Invalid ST completion speaker names")

        shared = {
            "style": "st_instruct_v1",
            "context_preset": preset["context_preset"],
            "instruct_preset": preset["instruct_preset"],
        }

        if "story_values" in preset:
            shared["story_values"] = preset["story_values"]

        npc_format = {
            **shared,
            "user_name": player_name,
            "character_name": npc_name,
        }
        player_format = {
            **shared,
            "user_name": npc_name,
            "character_name": player_name,
        }

        # Fail before creating a run if this preset is outside
        # our supported text-only SillyTavern subset.
        for fmt in (npc_format, player_format):
            render_st_completion_prompt(
                [{"role": "system", "content": "Validation context"}],
                fmt,
            )

        provenance = {
            "id": preset["id"],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content": preset,
        }

        return npc_format, player_format, provenance

    if preset.get("style") == "native_turns_v1":
        required = [
            "begin", "turn_start", "role_separator",
            "turn_end", "user_role", "assistant_role",
            "system_handling",
        ]
        if preset.get("system_handling") == "native":
            required.append("system_role")

        for key in required:
            if not isinstance(preset.get(key), str):
                raise ValueError(f"Invalid native template field: {key}")

        if preset["system_handling"] not in {"native", "first_user"}:
            raise ValueError("Unsupported system handling")

        shared = {
            "style": preset["style"],
            **{key: preset[key] for key in required},
        }

        if "system_suffix" in preset:
            if not isinstance(preset["system_suffix"], str):
                raise ValueError("Invalid system_suffix")
            shared["system_suffix"] = preset["system_suffix"]

        if "response_parser" in preset:
            if preset["response_parser"] not in {
                "atem_v1", "atem_direct_v1",
                "atem_thinking_prefill_v1"
            }:
                raise ValueError("Unsupported completion response parser")
            shared["response_parser"] = preset["response_parser"]

        if "generation_suffix" in preset:
            if not isinstance(preset["generation_suffix"], str):
                raise ValueError("Invalid generation_suffix")
            shared["generation_suffix"] = preset["generation_suffix"]

        provenance = {
            "id": preset["id"],
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content": preset,
        }
        return dict(shared), dict(shared), provenance

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
