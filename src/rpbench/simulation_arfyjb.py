"""Snapshot-based ArfyJB rendering for NPC self-play and resume."""

from pathlib import Path

from .prompt_chat import validate_strict_chat_template
from .refusal_arfyjb import load_arfyjb_source


def load_arfyjb_npc(runtime, config_path):
    settings = runtime.get("arfyjb_npc")
    if settings is None:
        return None

    if not isinstance(settings, dict) or set(settings) != {"source"}:
        raise ValueError("arfyjb_npc requires exactly source")

    reference = settings["source"]
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Invalid ArfyJB source")

    source_path = (
        Path(config_path).resolve().parent / reference
    ).resolve()

    payload, source_hash = load_arfyjb_source(source_path)

    if payload.get("benchmark_defaults") != {
        "main": True,
        "post": True,
        "cyoa": False,
        "final": False,
        "assistant_prefill": False,
    }:
        raise ValueError("Unsupported ArfyJB benchmark settings")

    return {
        "mode": "arfyjb_universal",
        "source_sha256": source_hash,
        "source_path": str(source_path),
        "source": payload,
        "enabled_components": ["main", "post"],
        "exact_sillytavern_serialization": False,
    }


def render_arfyjb_npc(messages, archived):
    if archived.get("mode") != "arfyjb_universal":
        raise ValueError("Invalid ArfyJB snapshot")

    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[0].get("role") != "system"
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("Expected NPC system and final user turn")

    components = archived["source"]["components"]
    main = components["main"]["content"]
    post = components["post"]["content"]

    result = [
        {
            "role": "system",
            "content": (
                main
                + "\n\n<rpbench-scenario-system>\n"
                + messages[0]["content"]
                + "\n</rpbench-scenario-system>"
            ),
        },
        *[dict(message) for message in messages[1:-1]],
        {
            "role": "user",
            "content": (
                "<rpbench-arfyjb-post-instruction>\n"
                + post
                + "\n</rpbench-arfyjb-post-instruction>\n\n"
                + messages[-1]["content"]
            ),
        },
    ]

    validate_strict_chat_template(result)
    return result
