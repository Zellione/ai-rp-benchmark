"""Resumable, provenance-aware SillyTavern preset refusal batches."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .core import dump
from .deployment_preflight import get_json, verify
from .refusal_batch import load_plan
from .refusal_probe import run_probe


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def experiment_identity(
    model_identity,
    config_sha256,
    preset_id,
    preset_sha256,
    character_id,
    compatibility,
):
    return {
        "model_identity": model_identity,
        "config_sha256": config_sha256,
        "preset_id": preset_id,
        "preset_sha256": preset_sha256,
        "character_id": character_id,
        "backend_compat": compatibility,
    }


def experiment_directory(root, identity):
    fingerprint = digest(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))

    model = identity["model_identity"]

    name = safe_name(
        f"{model['id']}__{model['variant']}__"
        f"{model['quantization']}__{identity['preset_id']}"
    )

    return Path(root) / f"{name}__{fingerprint[:16]}"


def validate_archive(directory, identity, plan):
    manifest_path = directory / "experiment.json"

    if directory.exists():
        if not manifest_path.exists():
            if any(directory.iterdir()):
                raise ValueError(
                    "Existing batch lacks experiment manifest"
                )
        elif json.loads(
            manifest_path.read_text(encoding="utf-8")
        ) != identity:
            raise ValueError("Experiment identity mismatch")

    expected = {
        item["id"]: item["scenario_sha256"]
        for item in plan
    }

    archived = {}

    for path in sorted(directory.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row.get("scenario_id")

        if scenario not in expected:
            raise ValueError(f"Unexpected scenario: {scenario}")

        if row.get("scenario_sha256") != expected[scenario]:
            raise ValueError(f"Scenario fingerprint mismatch: {path}")

        if row.get("config_sha256") != identity["config_sha256"]:
            raise ValueError(f"Config fingerprint mismatch: {path}")

        if row.get("model_identity") != identity["model_identity"]:
            raise ValueError(f"Model identity mismatch: {path}")

        preset = row.get("preset") or {}
        if (
            preset.get("id") != identity["preset_id"]
            or preset.get("source_sha256")
            != identity["preset_sha256"]
            or preset.get("character_id")
            != identity["character_id"]
        ):
            raise ValueError(f"Preset identity mismatch: {path}")

        actual_compat = (
            (row.get("preset_backend_compat") or {}).get("strategy")
        )
        if actual_compat != identity["backend_compat"]:
            raise ValueError(f"Backend compatibility mismatch: {path}")

        archived.setdefault(scenario, []).append(row)

    return archived


def pending_cases(plan, archived, retry_failed):
    pending = []

    for item in plan:
        attempts = archived.get(item["id"], [])

        if any(row.get("status") == "complete" for row in attempts):
            continue

        if attempts and not retry_failed:
            continue

        pending.append(item)

    return pending


def run_preset_batch(
    manifest,
    config,
    preset,
    output,
    *,
    character_id=100001,
    compatibility=None,
    retry_failed=False,
):
    plan, model_identity, config_sha256 = load_plan(
        manifest, config
    )

    preset = Path(preset)
    preset_sha256 = digest(preset.read_bytes())
    preset_id = preset.stem

    identity = experiment_identity(
        model_identity,
        config_sha256,
        preset_id,
        preset_sha256,
        character_id,
        compatibility,
    )

    directory = experiment_directory(output, identity)
    archived = validate_archive(directory, identity, plan)
    pending = pending_cases(plan, archived, retry_failed)

    print("Experiment:", directory.name)
    print("Preset SHA256:", preset_sha256)
    print("Planned:", len(plan))
    print("Pending:", len(pending))

    # Perform live verification before creating a new experiment.
    if pending:
        settings = json.loads(
            Path(config).read_text(encoding="utf-8")
        )
        backend = settings["backend"]

        if backend["name"] != "mock":
            base = backend["base_url"].rstrip("/")
            models = get_json(base + "/v1/models")
            props = (
                get_json(base + "/props")
                if backend["name"] == "llamacpp"
                else None
            )
            verification = verify(settings, models, props)
            print("Preflight:", json.dumps(verification))

    directory.mkdir(parents=True, exist_ok=True)

    manifest_path = directory / "experiment.json"
    if not manifest_path.exists():
        dump(manifest_path, identity)

    failures = 0

    for index, item in enumerate(pending, 1):
        print(
            f"{index:02d}/{len(pending):02d} | {item['id']}",
            flush=True,
        )

        row, _ = run_probe(
            item["file"],
            config,
            directory,
            preset_path=preset,
            preset_character_id=character_id,
            preset_backend_compat=compatibility,
        )

        if row["status"] != "complete":
            failures += 1
            print(
                "Unsuccessful:",
                row.get("error", row["status"]),
                flush=True,
            )

    print(
        "Finished:",
        len(pending) - failures,
        "complete,",
        failures,
        "unsuccessful",
    )
    print("Archive:", directory)

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--character-id", type=int, default=100001)
    parser.add_argument(
        "--backend-compat",
        choices=["alternating-chat-v1"],
    )
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    failures = run_preset_batch(
        args.manifest,
        args.config,
        args.preset,
        args.out,
        character_id=args.character_id,
        compatibility=args.backend_compat,
        retry_failed=args.retry_failed,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
