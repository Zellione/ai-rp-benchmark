"""Unified, resumable refusal experiment executor."""

import argparse
import hashlib
import json
from pathlib import Path

from .core import dump
from .deployment_preflight import get_json, verify
from .refusal_batch import load_plan
from .refusal_experiment import load_matrix, validate_matrix
from .refusal_probe import run_probe


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(raw)


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


ADAPTER_FILES = {
    "baseline": [
        "refusal_probe.py",
    ],
    "sillytavern": [
        "refusal_probe.py",
        "preset_adapter.py",
        "prompt_st_chat.py",
        "prompt_presets.py",
        "prompt_injections.py",
        "prompt_chat.py",
        "st_template_engine.py",
        "preset_backend_compat.py",
    ],
    "megumin-v9": [
        "refusal_probe.py",
        "refusal_megumin.py",
        "megumin_v9.py",
        "megumin_v9_adapter.py",
        "preset_inhistory_compat.py",
        "preset_backend_compat.py",
        "preset_adapter.py",
        "prompt_st_chat.py",
        "prompt_chat.py",
        "st_template_engine.py",
    ],
    "arfyjb": [
        "refusal_probe.py",
        "refusal_arfyjb.py",
        "prompt_chat.py",
    ],
}

SHARED_FILES = [
    "core.py",
    "sampling.py",
    "prompt_models.py",
]


def implementation_hashes(adapter):
    base = Path(__file__).resolve().parent
    names = sorted(set(SHARED_FILES + ADAPTER_FILES[adapter]))

    return {
        name: file_sha256(base / name)
        for name in names
    }


def preset_inputs(entry):
    kind = entry["adapter"]
    paths = {}

    if kind in {"sillytavern", "arfyjb"}:
        paths["preset_source"] = entry["source"]

    elif kind == "megumin-v9":
        config_path = Path(entry["config"])
        profile = json.loads(config_path.read_text(encoding="utf-8"))
        directory = Path(entry["engine_directory"])

        paths = {
            "megumin_profile": config_path,
            "preset_source": profile["source"]["preset"],
            "megumin_engine": directory / "engine.json",
            "megumin_cot": directory / "cot.json",
        }

    return {name: file_sha256(path) for name, path in paths.items()}


def build_identity(model, preset, config, config_digest):
    return {
        "schema_version": 1,
        "runner": "refusal-experiment-v1",
        "model_key": model["id"],
        "model_identity": config["model_identity"],
        "model_config_sha256": config_digest,
        "output_tokens": config["sampling"]["max_tokens"],
        "preset_key": preset["id"],
        "adapter": preset["adapter"],
        "adapter_options": {
            key: value
            for key, value in preset.items()
            if key not in {"id", "adapter", "source", "config",
                           "engine_directory"}
        },
        "preset_inputs_sha256": preset_inputs(preset),
        "implementation_sha256": implementation_hashes(
            preset["adapter"]
        ),
    }


def experiment_dir(root, identity):
    name = (
        identity["model_key"] + "__" +
        identity["preset_key"] + "__" +
        str(identity["output_tokens"]) + "tok"
    )
    safe = "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in name
    )
    return Path(root) / f"{safe}__{fingerprint(identity)[:16]}"


def validate_archive(directory, identity, plan):
    manifest = directory / "experiment.json"
    if directory.exists():
        if not manifest.exists():
            if any(directory.iterdir()):
                raise ValueError("Experiment has no identity manifest")
        elif json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Archived experiment identity mismatch")

    expected = {
        case["id"]: case["scenario_sha256"]
        for case in plan
    }
    archived = {}

    for path in sorted(directory.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row.get("scenario_id")

        if scenario not in expected:
            raise ValueError(f"Unknown archived scenario: {path}")
        if row.get("scenario_sha256") != expected[scenario]:
            raise ValueError(f"Scenario fingerprint mismatch: {path}")
        if row.get("config_sha256") != identity["model_config_sha256"]:
            raise ValueError(f"Model config fingerprint mismatch: {path}")
        if row.get("model_identity") != identity["model_identity"]:
            raise ValueError(f"Model identity mismatch: {path}")

        preset = row.get("preset")
        if identity["adapter"] == "baseline":
            if preset is not None:
                raise ValueError(f"Unexpected baseline preset: {path}")
        else:
            if not isinstance(preset, dict):
                raise ValueError(f"Missing preset metadata: {path}")

            expected_sha = identity["preset_inputs_sha256"].get(
                "preset_source"
            )
            if preset.get("source_sha256") != expected_sha:
                raise ValueError(f"Preset source mismatch: {path}")

            for field, input_key in (
                ("config_sha256", "megumin_profile"),
                ("engine_sha256", "megumin_engine"),
                ("cot_sha256", "megumin_cot"),
            ):
                if input_key in identity["preset_inputs_sha256"]:
                    if preset.get(field) != (
                        identity["preset_inputs_sha256"][input_key]
                    ):
                        raise ValueError(f"{field} mismatch: {path}")

        archived.setdefault(scenario, []).append(row)

    return archived


def select_pending(plan, archived, retry_failed=False):
    pending = []

    for case in plan:
        attempts = archived.get(case["id"], [])

        if any(r.get("status") == "complete" for r in attempts):
            continue
        if attempts and not retry_failed:
            continue

        pending.append(case)

    return pending


def probe_options(preset):
    kind = preset["adapter"]

    if kind == "baseline":
        return {}

    if kind == "sillytavern":
        return {
            "preset_path": preset["source"],
            "preset_character_id": preset.get("character_id", 100001),
            "preset_backend_compat": preset["backend_compat"],
        }

    if kind == "megumin-v9":
        return {
            "megumin_config": preset["config"],
            "megumin_engine_directory": preset["engine_directory"],
        }

    if kind == "arfyjb":
        return {"arfyjb_source": preset["source"]}

    raise ValueError(f"Unsupported adapter: {kind}")


def verify_backend(config):
    backend = config["backend"]

    if backend["name"] == "mock":
        return {"backend": "mock"}

    base = backend["base_url"].rstrip("/")
    models = get_json(base + "/v1/models")
    props = (
        get_json(base + "/props")
        if backend["name"] == "llamacpp"
        else None
    )

    return verify(config, models, props)


def execute(
    matrix_path,
    model_key,
    *,
    preset_key=None,
    output="/scratch/rpbench-presets/unified-8k",
    limit=None,
    retry_failed=False,
    dry_run=False,
):
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)

    models = [
        item for item in matrix["models"]
        if item["id"] == model_key
    ]
    if len(models) != 1:
        raise ValueError(f"Unknown model: {model_key}")

    selected_presets = [
        item for item in matrix["presets"]
        if preset_key is None or item["id"] == preset_key
    ]
    if not selected_presets:
        raise ValueError(f"Unknown preset: {preset_key}")

    model = models[0]
    config_path = Path(model["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    plan, _, config_sha = load_plan(
        matrix["scenario_manifest"], config_path
    )

    options = []

    for preset in selected_presets:
        identity = build_identity(model, preset, config, config_sha)
        directory = experiment_dir(output, identity)
        archived = validate_archive(directory, identity, plan)
        pending = select_pending(plan, archived, retry_failed)

        options.append((preset, identity, directory, pending))

    total_pending = sum(len(row[3]) for row in options)

    print("=== UNIFIED EXPERIMENT ===")
    print("Model:", model_key)
    print("Output tokens:", matrix["output_tokens"])
    print("Presets:", len(options))
    print("Pending scenarios:", total_pending)

    for preset, _, directory, pending in options:
        print(
            f"  {preset['id']}: pending={len(pending)} "
            f"archive={directory.name}"
        )

    if dry_run:
        print("Dry run: no server requests or archive writes")
        return 0

    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    if total_pending:
        print("Preflight:", json.dumps(verify_backend(config)))

    remaining = limit
    failures = 0

    for preset, identity, directory, pending in options:
        if remaining == 0:
            break

        selected = pending if remaining is None else pending[:remaining]
        if not selected:
            continue

        directory.mkdir(parents=True, exist_ok=True)
        manifest_file = directory / "experiment.json"

        if not manifest_file.exists():
            dump(manifest_file, identity)

        kwargs = probe_options(preset)

        for case in selected:
            print(
                f"{preset['id']} | {case['id']}",
                flush=True,
            )

            row, _ = run_probe(
                case["file"],
                config_path,
                directory,
                **kwargs,
            )

            if row["status"] != "complete":
                failures += 1

        if remaining is not None:
            remaining -= len(selected)

    print("Unsuccessful attempts:", failures)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--preset")
    parser.add_argument(
        "--out",
        default="/scratch/rpbench-presets/unified-8k",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failures = execute(
        args.matrix,
        args.model,
        preset_key=args.preset,
        output=args.out,
        limit=args.limit,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
