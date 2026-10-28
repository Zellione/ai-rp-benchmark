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


def run_probe(scenario_path, config_path, output_dir):
    scenario_path = Path(scenario_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    scenario_bytes = scenario_path.read_bytes()
    config_bytes = config_path.read_bytes()

    scenario = json.loads(scenario_bytes)
    config = json.loads(config_bytes)

    messages = build_messages(scenario)
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
    args = parser.parse_args()

    record, _ = run_probe(
        args.scenario, args.config, args.out
    )
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
