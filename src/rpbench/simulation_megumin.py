"""Archived historical Megumin V9 renderer for simulations and resume."""

import copy
import hashlib
import json
from pathlib import Path

from .megumin_v9 import make_profile_resolver
from .megumin_v9_adapter import MeguminV9PresetAdapter
from .preset_adapter import PresetContext
from .preset_backend_compat import lower_to_alternating_chat
from .preset_inhistory_compat import relocate_prefinal_system
from .prompt_chat import validate_strict_chat_template
from .prompt_models import PromptMessage
from .prompt_presets import SillyTavernChatPreset


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_file(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), digest(raw)


def load_megumin_npc(runtime, config_path, npc_system):
    settings = runtime.get("megumin_npc")
    if not isinstance(settings, dict):
        raise ValueError("megumin_npc must be an object")

    if set(settings) != {
        "config", "engine_directory", "character_name", "user_name"
    }:
        raise ValueError("Invalid megumin_npc settings")

    base = Path(config_path).resolve().parent
    profile_path = (base / settings["config"]).resolve()
    engine_dir = (base / settings["engine_directory"]).resolve()

    adapter = MeguminV9PresetAdapter(profile_path, engine_dir)

    profile, profile_hash = read_file(profile_path)
    source, source_hash = read_file(adapter.source)
    engine, engine_hash = read_file(engine_dir / "engine.json")
    cot, cot_hash = read_file(engine_dir / "cot.json")

    character_id = adapter.config["character_id"]
    ordered = adapter.preset.ordered_prompts(character_id)

    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"]
        and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }

    if "charDescription" not in markers:
        raise ValueError("Missing Megumin charDescription marker")

    markers["charDescription"] = npc_system

    names = {}
    for key in ("character_name", "user_name"):
        value = settings[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid megumin_npc.{key}")
        names[key] = value

    snapshot = {
        "mode": "megumin_v9",
        "profile": profile,
        "source": source,
        "engine": engine,
        "cot": cot,
        "sha256": {
            "profile": profile_hash,
            "source": source_hash,
            "engine": engine_hash,
            "cot": cot_hash,
        },
        "character_id": character_id,
        "markers": markers,
        **names,
        "unsupported_features": [
            "dynamic_npc_bank",
            "extension_memory",
            "story_planner",
            "assistant_prefill",
        ],
        "exact_sillytavern_serialization": False,
    }

    # Preflight rendering before any simulation archive is created.
    render_megumin_npc(
        [
            {"role": "system", "content": npc_system},
            {"role": "user", "content": "Validation turn"},
        ],
        snapshot,
    )

    return snapshot


def render_megumin_npc(messages, archived):
    if archived.get("mode") != "megumin_v9":
        raise ValueError("Invalid Megumin renderer")

    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[0].get("role") != "system"
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("Expected NPC system and final user turn")

    profile = archived["profile"]
    source = archived["source"]

    original_source_hash = profile["source"]["preset_sha256"]
    if archived["sha256"]["source"] != original_source_hash:
        raise ValueError("Megumin source hash mismatch")

    # Reconstruct the existing adapter from archived data only.
    adapter = object.__new__(MeguminV9PresetAdapter)
    adapter.config = copy.deepcopy(profile)
    adapter.engine = copy.deepcopy(archived["engine"])
    adapter.cot = copy.deepcopy(archived["cot"])
    adapter.source_sha256 = archived["sha256"]["source"]
    adapter.preset_id = "megumin-v9-mirage"
    adapter.preset = SillyTavernChatPreset(
        source, allow_disabled_orphans=True
    )
    adapter.resolver = make_profile_resolver(
        adapter.engine, adapter.cot, adapter.config
    )

    ordered = adapter.preset.ordered_prompts(
        archived["character_id"]
    )
    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"]
        and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }
    markers["charDescription"] = messages[0]["content"]

    history = tuple(
        PromptMessage(item["role"], item["content"])
        for item in messages[1:]
    )

    result = adapter.render(PresetContext(
        history=history,
        markers=markers,
        character_name=archived["character_name"],
        user_name=archived["user_name"],
        character_id=archived["character_id"],
    ))

    prepared, _ = relocate_prefinal_system(result.messages)
    lowered, _ = lower_to_alternating_chat(prepared)
    validate_strict_chat_template(lowered)
    return lowered
