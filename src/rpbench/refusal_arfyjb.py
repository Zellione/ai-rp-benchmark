"""Auditable ArfyJB universal prompt adapter for refusal benchmarks."""

import hashlib
import json
from pathlib import Path

from .prompt_chat import validate_strict_chat_template
from .refusal_probe import build_messages


class ArfyJBConfigurationError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_arfyjb_source(path):
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)

    if payload.get("id") != "arfyjb-universal-source":
        raise ArfyJBConfigurationError("Unexpected source ID")

    components = payload.get("components")
    if not isinstance(components, dict):
        raise ArfyJBConfigurationError("Missing components")

    for key in ("main", "post", "cyoa", "final"):
        entry = components.get(key)
        if not isinstance(entry, dict):
            raise ArfyJBConfigurationError(f"Missing component: {key}")

        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ArfyJBConfigurationError(f"Empty component: {key}")

        if sha256(content.encode("utf-8")) != entry.get("sha256"):
            raise ArfyJBConfigurationError(
                f"Component fingerprint mismatch: {key}"
            )

    return payload, sha256(raw)


def build_arfyjb_messages(scenario, *, source_path):
    """Render main + original fixed history + prefinal post-instruction.

    CYOA, Final Message and assistant prefill are explicitly disabled.
    """
    source, source_digest = load_arfyjb_source(source_path)
    defaults = source.get("benchmark_defaults", {})

    required = {
        "main": True,
        "post": True,
        "cyoa": False,
        "final": False,
        "assistant_prefill": False,
    }

    if defaults != required:
        raise ArfyJBConfigurationError(
            "Unsupported ArfyJB benchmark module selection"
        )

    baseline = build_messages(scenario)
    history = baseline[1:-1]
    final_turn = baseline[-1]

    main = source["components"]["main"]["content"]
    post = source["components"]["post"]["content"]

    # Keep the scenario system instructions, but distinguish them
    # from the upstream ArfyJB main prompt.
    system = (
        main
        + "\n\n<rpbench-scenario-system>\n"
        + baseline[0]["content"]
        + "\n</rpbench-scenario-system>"
    )

    # Pre-final instructions are attached to the final user turn
    # to preserve strict user/assistant alternation.
    final_content = (
        "<rpbench-arfyjb-post-instruction>\n"
        + post
        + "\n</rpbench-arfyjb-post-instruction>\n\n"
        + final_turn["content"]
    )

    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": final_content},
    ]

    validate_strict_chat_template(messages)

    # Every original chat turn must remain unchanged and in order.
    assert messages[1:-1] == history
    assert messages[-1]["content"].endswith(final_turn["content"])

    metadata = {
        "id": "arfyjb-universal",
        "source_sha256": source_digest,
        "source_url": source["source_url"],
        "source_html_sha256": source["source_html_sha256"],
        "source_version_verified": False,
        "component_sha256": {
            key: source["components"][key]["sha256"]
            for key in ("main", "post", "cyoa", "final")
        },
        "enabled_components": ["main", "post"],
        "disabled_components": [
            "cyoa", "final", "assistant_prefill"
        ],
        "backend_compat": "arfyjb-prefinal-user-v1",
        "exact_sillytavern_serialization": False,
        "prefill_applied": False,
    }

    return messages, metadata
