"""Execute one fixed-history roleplay refusal probe."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Backend, dump
from .sampling import validate_sampling
from .preset_adapter import (
    SillyTavernPresetAdapter,
    render_with_adapter,
)
from .prompt_models import PromptMessage
from .prompt_chat import validate_strict_chat_template
from .preset_backend_compat import (
    lower_to_alternating_chat,
    lower_to_alternating_chat_v2,
)


def validate_scenario(scenario):
    required = {
        "id", "system", "turns", "fixed_history",
        "probe_turn", "category", "expected_behavior",
    }
    if not required <= set(scenario):
        raise ValueError("Missing refusal scenario fields")

    if scenario["expected_behavior"] not in {"continue", "boundary"}:
        raise ValueError("Invalid expected behavior")

    turns = scenario["turns"]
    history = scenario["fixed_history"]

    if (
        not isinstance(turns, list)
        or len(turns) != 3
        or scenario["probe_turn"] != 3
        or not isinstance(history, list)
        or len(history) != 4
    ):
        raise ValueError("Invalid fixed-history structure")

    expected_roles = ["user", "assistant", "user", "assistant"]
    for message, role in zip(history, expected_roles):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != role
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError("Invalid fixed-history message")

    if (
        history[0]["content"] != turns[0]
        or history[2]["content"] != turns[1]
    ):
        raise ValueError("Fixed history differs from scripted turns")

    if not isinstance(turns[2], str) or not turns[2].strip():
        raise ValueError("Missing final player probe")

    if not isinstance(scenario["system"], str) or not scenario["system"].strip():
        raise ValueError("Missing system instructions")


def build_messages(scenario):
    validate_scenario(scenario)
    return [
        {"role": "system", "content": scenario["system"]},
        *scenario["fixed_history"],
        {"role": "user", "content": scenario["turns"][2]},
    ]



def validate_model_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Missing model_identity")

    required = {
        "id", "base_model", "publisher",
        "variant", "quantization", "source",
    }
    if set(identity) != required:
        raise ValueError("Invalid model_identity fields")

    for key in required - {"source"}:
        value = identity[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid model_identity.{key}")

    source = identity["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"type", "path"}
        or source["type"] not in {"local", "huggingface"}
        or not isinstance(source["path"], str)
        or not source["path"].strip()
    ):
        raise ValueError("Invalid model_identity.source")

    return identity


def build_preset_messages(
    scenario,
    preset_path,
    character_id=100001,
):
    # The scenario system text is supplied through charDescription,
    # rather than being duplicated in the chat history.
    original = build_messages(scenario)
    history = [
        PromptMessage(row["role"], row["content"])
        for row in original[1:]
    ]

    adapter = SillyTavernPresetAdapter(
        preset_path,
        preset_id=Path(preset_path).stem,
        allow_disabled_orphans=True,
    )

    ordered = adapter.preset.ordered_prompts(character_id)
    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"]
        and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }

    if "charDescription" not in markers:
        raise ValueError(
            "Preset lacks active charDescription marker"
        )
    markers["charDescription"] = scenario["system"]

    result = render_with_adapter(
        adapter,
        history,
        markers=markers,
        character_name="Mara Voss",
        user_name="Player",
        character_id=character_id,
    )

    # History may be followed by post-history system instructions.
    expected = [message.to_dict() for message in history]
    cursor = 0
    for message in result.messages:
        if cursor < len(expected) and message == expected[cursor]:
            cursor += 1

    if cursor != len(expected):
        raise ValueError("Preset altered fixed history")

    if any(
        "{{" in message["content"] or "}}" in message["content"]
        for message in result.messages
    ):
        raise ValueError("Unresolved template expressions")

    metadata = {
        "id": result.preset_id,
        "source_sha256": result.source_sha256,
        "character_id": character_id,
        "applied_hooks": result.applied_hooks,
        "unsupported_features": [
            "external_sillytavern_extensions",
            "tracker_regex_postprocessing",
        ],
    }
    return result.messages, metadata


def run_probe(
    scenario_path,
    config_path,
    output_dir,
    *,
    preset_path=None,
    preset_character_id=100001,
    preset_backend_compat=None,
    megumin_config=None,
    megumin_engine_directory=None,
    arfyjb_source=None,
):
    scenario_path = Path(scenario_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    scenario_bytes = scenario_path.read_bytes()
    config_bytes = config_path.read_bytes()

    scenario = json.loads(scenario_bytes)
    config = json.loads(config_bytes)

    preset_metadata = None
    preset_messages = None
    compatibility_metadata = None

    if arfyjb_source is not None:
        if any(value is not None for value in (
            preset_path,
            preset_backend_compat,
            megumin_config,
            megumin_engine_directory,
        )):
            raise ValueError("ArfyJB cannot be combined with another preset")

        from .refusal_arfyjb import build_arfyjb_messages

        messages, preset_metadata = build_arfyjb_messages(
            scenario, source_path=arfyjb_source
        )
    elif megumin_config is not None:
        if preset_path is not None or preset_backend_compat is not None:
            raise ValueError(
                "Megumin configuration cannot be combined with "
                "standard preset options"
            )
        if megumin_engine_directory is None:
            raise ValueError("Missing Megumin engine directory")

        from .refusal_megumin import build_megumin_messages

        messages, preset_metadata, preset_messages = (
            build_megumin_messages(
                scenario,
                config_path=megumin_config,
                engine_directory=megumin_engine_directory,
            )
        )
        compatibility_metadata = preset_metadata["backend_compat"]
    elif preset_path is None:
        messages = build_messages(scenario)
    else:
        messages, preset_metadata = build_preset_messages(
            scenario,
            preset_path,
            preset_character_id,
        )

    if preset_backend_compat is not None:
        if preset_path is None:
            raise ValueError(
                "Backend compatibility requires a preset"
            )
        strategies = {
            "alternating-chat-v1": lower_to_alternating_chat,
            "alternating-chat-v2": lower_to_alternating_chat_v2,
        }
        if preset_backend_compat not in strategies:
            raise ValueError(
                f"Unknown compatibility strategy: {preset_backend_compat}"
            )

        preset_messages = messages
        messages, compatibility_metadata = strategies[
            preset_backend_compat
        ](preset_messages)
        validate_strict_chat_template(messages)

    engine = config["backend"]
    model_identity = validate_model_identity(
        config["model_identity"]
    )
    sampling = config.get("sampling", {})
    prompt_format = config.get("prompt_format", {})

    if engine["name"] in {"ninfer", "strata"} and not engine.get(
        "verified_openai_api", False
    ):
        raise ValueError("Unverified OpenAI-compatible backend")

    backend = Backend(
        name=engine["name"],
        base_url=engine.get("base_url", ""),
        model=engine.get("model", "mock"),
        mode=engine.get("mode", "chat"),
        api_key=os.getenv(engine.get("api_key_env", "RPBENCH_API_KEY")),
        timeout=int(engine.get("timeout", 300)),
    )

    validate_sampling(backend.name, backend.mode, sampling)
    url, payload = backend.prepare(messages, sampling, prompt_format)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    run_dir = output_dir / (
        f"{timestamp}_{scenario['id']}_probe"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "schema_version": 1,
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "expected_behavior": scenario["expected_behavior"],
        "probe_turn": scenario["probe_turn"],
        "backend": {
            "name": backend.name,
            "model": backend.model,
            "mode": backend.mode,
        },
        "sampling": sampling,
        "model_identity": model_identity,
        "scenario_sha256": hashlib.sha256(
            scenario_bytes
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            config_bytes
        ).hexdigest(),
        "messages": messages,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if preset_metadata is not None:
        record["preset"] = preset_metadata

    if compatibility_metadata is not None:
        record["preset_messages"] = preset_messages
        record["preset_backend_compat"] = compatibility_metadata

    dump(run_dir / "result.json", record)
    started = time.perf_counter()

    try:
        content, response = backend.generate(url, payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        record.update({
            "response": content,
            "reasoning": (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            ),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "raw_response": response,
        })

        if record["finish_reason"] == "length":
            record["status"] = "truncated"
        elif not content.strip():
            record["status"] = "empty_response"
        else:
            record["status"] = "complete"

    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"

    finally:
        record["elapsed_seconds"] = round(
            time.perf_counter() - started, 3
        )
        record["finished_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )
        dump(run_dir / "result.json", record)

    print(
        f"{scenario['id']} | {backend.model} | "
        f"{record['status']} | "
        f"{record['elapsed_seconds']}s",
        flush=True,
    )
    print("Saved:", run_dir, flush=True)

    return record, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset")
    parser.add_argument("--megumin-config")
    parser.add_argument("--megumin-engine-directory")
    parser.add_argument("--arfyjb-source")
    parser.add_argument(
        "--preset-character-id",
        type=int,
        default=100001,
    )
    parser.add_argument(
        "--preset-backend-compat",
        choices=["alternating-chat-v1", "alternating-chat-v2"],
    )
    args = parser.parse_args()

    record, _ = run_probe(
        args.scenario,
        args.config,
        args.out,
        preset_path=args.preset,
        preset_character_id=args.preset_character_id,
        preset_backend_compat=args.preset_backend_compat,
        megumin_config=args.megumin_config,
        megumin_engine_directory=args.megumin_engine_directory,
        arfyjb_source=args.arfyjb_source,
    )
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
