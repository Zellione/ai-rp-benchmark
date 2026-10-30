"""Megumin V9 Mirage integration for fixed-history refusal scenarios."""

import hashlib
import json
from pathlib import Path

from .megumin_v9_adapter import MeguminV9PresetAdapter
from .preset_adapter import render_with_adapter
from .preset_backend_compat import (
    lower_to_alternating_chat,
    messages_sha256,
)
from .preset_inhistory_compat import relocate_prefinal_system
from .prompt_chat import validate_strict_chat_template
from .prompt_models import PromptMessage
from .refusal_probe import build_messages


def build_megumin_messages(
    scenario,
    *,
    config_path,
    engine_directory,
):
    adapter = MeguminV9PresetAdapter(
        config_path,
        engine_directory,
    )

    original = build_messages(scenario)
    history = [
        PromptMessage(row["role"], row["content"])
        for row in original[1:]
    ]

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
        raise ValueError("Missing charDescription marker")

    markers["charDescription"] = scenario["system"]

    result = render_with_adapter(
        adapter,
        history,
        markers=markers,
        character_name="Mara Voss",
        user_name="Player",
        character_id=character_id,
    )

    expected = [m.to_dict() for m in history]
    cursor = 0

    for message in result.messages:
        if cursor < len(expected) and message == expected[cursor]:
            cursor += 1

    if cursor != len(expected):
        raise ValueError("Original fixed history not preserved")

    prepared, relocated = relocate_prefinal_system(
        result.messages
    )
    lowered, compatibility = lower_to_alternating_chat(prepared)
    validate_strict_chat_template(lowered)

    original_text = "\n".join(
        m["content"] for m in result.messages
    )
    lowered_text = "\n".join(
        m["content"] for m in lowered
    )

    for message in result.messages:
        if message["content"] not in lowered_text:
            raise ValueError("Original prompt content lost")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    engine_dir = Path(engine_directory)
    config_file = Path(config_path)

    metadata = {
        "id": result.preset_id,
        "source_sha256": result.source_sha256,
        "config_sha256": digest(config_file),
        "engine_sha256": digest(engine_dir / "engine.json"),
        "cot_sha256": digest(engine_dir / "cot.json"),
        "character_id": character_id,
        "applied_hooks": [
            *result.applied_hooks,
            "relocate_prefinal_system",
            "alternating_chat_lowering",
        ],
        "prefinal_system_relocated": relocated,
        "backend_compat": compatibility,
        "original_messages_sha256": messages_sha256(
            result.messages
        ),
        "sent_messages_sha256": messages_sha256(lowered),
        "prefill_applied": False,
        "unsupported_features": result.unsupported_features,
        "exact_sillytavern_serialization": False,
    }

    return lowered, metadata, result.messages
