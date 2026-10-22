"""An intentionally minimal, auditable RP benchmark execution core."""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .sampling import validate_sampling


def dump(path: Path, value: Any) -> None:
    """Atomically replace a JSON file on the same filesystem."""
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False) + chr(10)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)
        temp_path = None

        # Persist the directory entry as well as the file data.
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def request_json(url: str, payload: dict, api_key: str | None, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def raw_prompt(messages: list[dict], format: dict) -> str:
    # Simple user-owned format, deliberately not advertised as a model chat template.
    prefixes = format.get("prefixes", {"system": "System: ", "user": "User: ", "assistant": "Assistant: "})
    sep = format.get("separator", "\n\n")
    suffix = format.get("assistant_prefix", prefixes["assistant"])
    return sep.join(prefixes[m["role"]] + m["content"] for m in messages) + sep + suffix


def roleplay_raw_prompt(messages: list[dict], fmt: dict) -> str:
    """Render a named-speaker completion prompt for one role."""
    if fmt.get("style") != "rp_dialogue_v1":
        raise ValueError("Unknown roleplay completion format")

    speakers = fmt.get("speakers")
    if (
        not isinstance(speakers, dict)
        or any(
            not isinstance(speakers.get(key), str)
            or not speakers[key].strip()
            for key in ("user", "assistant")
        )
    ):
        raise ValueError("Missing completion speakers")

    separator = fmt.get("separator", "\n\n")
    system_prefix = fmt.get("system_prefix", "Instructions:\n")
    scene_label = fmt.get("scene_label", "Scene")
    opening_as_scene = fmt.get("opening_as_scene", False)

    if not all(
        isinstance(value, str)
        for value in (separator, system_prefix, scene_label)
    ):
        raise ValueError("Invalid completion format strings")

    if not isinstance(opening_as_scene, bool):
        raise ValueError("Invalid opening_as_scene")

    parts = []
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        if role == "system":
            parts.append(system_prefix + content)
        elif role in {"user", "assistant"}:
            label = speakers[role]
            if (
                role == "user"
                and opening_as_scene
                and index == 1
            ):
                label = scene_label

            parts.append(f"{label}:\n{content}")
        else:
            raise ValueError(f"Unsupported message role: {role}")

    # Always end at the next speaker's generation boundary.
    parts.append(f'{speakers["assistant"]}:\n')
    return separator.join(parts)


@dataclass
class Backend:
    name: str
    base_url: str
    model: str
    mode: str
    api_key: str | None = None
    timeout: int = 300

    def prepare(self, messages: list[dict], sampling: dict, prompt_format: dict) -> tuple[str, dict]:
        validate_sampling(self.name, self.mode, sampling)
        root = self.base_url.rstrip("/")
        if self.name == "mock":
            return "mock://generate", {"messages": messages, "sampling": sampling}
        if self.mode == "chat":
            if self.name == "koboldcpp-native":
                raise ValueError("koboldcpp-native supports only completion mode")
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **sampling,
            }
            if self.name == "vllm" and "chat_template_kwargs" in prompt_format:
                payload["chat_template_kwargs"] = prompt_format["chat_template_kwargs"]
            return root + "/v1/chat/completions", payload
        if self.mode != "completion":
            raise ValueError(f"Unsupported mode: {self.mode}")
        if prompt_format.get("style") == "rp_dialogue_v1":
            prompt = roleplay_raw_prompt(messages, prompt_format)
        else:
            prompt = raw_prompt(messages, prompt_format)
        if self.name == "koboldcpp-native":
            native = dict(sampling)
            if "max_tokens" in native:
                native["max_length"] = native.pop("max_tokens")
            return root + "/api/v1/generate", {"prompt": prompt, **native}
        if self.name == "llamacpp-native":
            native = dict(sampling)
            if "max_tokens" in native:
                native["n_predict"] = native.pop("max_tokens")
            return root + "/completion", {"prompt": prompt, **native}
        return root + "/v1/completions", {"model": self.model, "prompt": prompt, "stream": False, **sampling}

    def generate(self, url: str, payload: dict) -> tuple[str, dict]:
        if self.name == "mock":
            turn = sum(m["role"] == "user" for m in payload["messages"])
            return f"[mock answer for turn {turn}]", {"mock": True}
        obj = request_json(url, payload, self.api_key, self.timeout)
        if self.name == "koboldcpp-native":
            answer = obj["results"][0]["text"]
        elif self.name == "llamacpp-native":
            answer = obj["content"]
        elif self.mode == "chat":
            answer = obj["choices"][0]["message"].get("content") or ""
        else:
            answer = obj["choices"][0]["text"]
        if not isinstance(answer, str):
            raise ValueError("Backend did not return string text")
        return answer, obj


def run(scenario_path: str, config_path: str, output: str) -> Path:
    scenario, config = load(scenario_path), load(config_path)
    engine = config["backend"]
    backend = Backend(engine["name"], engine.get("base_url", ""), engine.get("model", "mock"),
                      engine.get("mode", "chat"), os.getenv(engine.get("api_key_env", "RPBENCH_API_KEY")),
                      int(engine.get("timeout", 300)))
    supported = {"mock", "vllm", "llamacpp", "llamacpp-native", "koboldcpp", "koboldcpp-native", "ninfer", "strata"}
    if backend.name not in supported:
        raise ValueError(f"Unknown backend: {backend.name}")
    # Generic protocol route != validated engine-specific feature support.
    if backend.name in {"ninfer", "strata"} and not engine.get("verified_openai_api", False):
        raise ValueError(f"{backend.name} adapter is not verified; set verified_openai_api only after checking its server")
    sampling = config.get("sampling", {})
    validate_sampling(backend.name, backend.mode, sampling)
    fmt = config.get("prompt_format", {})
    turns = scenario["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("Scenario requires at least one user turn")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", scenario.get("id", "scenario"))
    run_dir = Path(output) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "created_utc": timestamp,
        "status": "running",
        "backend": {k: v for k, v in engine.items() if k != "api_key"},
        "sampling": sampling,
        "prompt_format": fmt,
        "scenario": scenario,
        "scenario_sha256": hashlib.sha256(Path(scenario_path).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "notes": "No token-level parity or context-length enforcement is claimed in this MVP.",
    }
    dump(run_dir / "manifest.json", manifest)
    messages = [{"role": "system", "content": scenario["system"]}]
    markdown = run_dir / "conversation.md"
    markdown.write_text(f"# {scenario.get('title', name)}\n\n## System\n\n{scenario['system']}\n\n", encoding="utf-8")
    try:
        for idx, user_text in enumerate(turns, 1):
            messages.append({"role": "user", "content": user_text})
            url, payload = backend.prepare(messages, sampling, fmt)
            started = time.perf_counter()
            append(run_dir / "events.jsonl", {"type": "request", "turn": idx, "endpoint": url, "payload": payload})
            try:
                answer, response = backend.generate(url, payload)
            except Exception as error:
                append(run_dir / "events.jsonl", {"type": "error", "turn": idx, "error": repr(error)})
                raise
            elapsed = time.perf_counter() - started
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            reasoning = message.get("reasoning")
            finish_reason = choice.get("finish_reason")
            response_status = (
                "reasoning_only" if not answer.strip() and reasoning
                else "empty_response" if not answer.strip()
                else "truncated" if finish_reason == "length"
                else "complete"
            )
            append(run_dir / "events.jsonl", {
                "type": "response",
                "turn": idx,
                "text": answer,
                "reasoning": reasoning,
                "response_status": response_status,
                "finish_reason": finish_reason,
                "usage": response.get("usage"),
                "elapsed_seconds": elapsed,
                "raw_response": response,
            })
            if response_status in {"reasoning_only", "empty_response"}:
                raise ValueError(
                    f"Turn {idx}: no visible assistant response "
                    f"({response_status}); raw response archived"
                )
            with markdown.open("a", encoding="utf-8") as f:
                f.write(f"## Turn {idx}\n\n### User\n\n{user_text}\n\n### Assistant\n\n{answer}\n\n")
                f.flush()
                os.fsync(f.fileno())
            messages.append({"role": "assistant", "content": answer})
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        dump(run_dir / "manifest.json", manifest)
    return run_dir
