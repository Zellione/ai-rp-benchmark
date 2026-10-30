"""Unified refusal experiment matrix validation and dry-run planning."""

import argparse
import hashlib
import json
from pathlib import Path

from .refusal_probe import validate_scenario, validate_model_identity
from .sampling import validate_sampling


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_matrix(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("schema_version") != 1:
        raise ValueError("Unsupported experiment matrix schema")
    if not data.get("models") or not data.get("presets"):
        raise ValueError("Missing models or presets")
    if type(data.get("output_tokens")) is not int or data["output_tokens"] <= 0:
        raise ValueError("Invalid output token budget")

    return data


def validate_matrix(data):
    manifest = Path(data["scenario_manifest"])
    scenarios = json.loads(manifest.read_text(encoding="utf-8"))

    if scenarios.get("schema_version") != 1:
        raise ValueError("Invalid scenario manifest")

    case_hashes = {}
    for item in scenarios["cases"]:
        path = manifest.parent / item["file"]
        scenario = json.loads(path.read_text(encoding="utf-8"))
        validate_scenario(scenario)

        if scenario["id"] != item["id"]:
            raise ValueError("Scenario ID mismatch")
        if scenario["id"] in case_hashes:
            raise ValueError("Duplicate scenario")

        case_hashes[scenario["id"]] = digest(path)

    model_hashes = {}
    for model in data["models"]:
        config_path = Path(model["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_model_identity(config["model_identity"])

        sampling = config.get("sampling", {})
        if sampling.get("max_tokens") != data["output_tokens"]:
            raise ValueError(
                f"{model['id']}: expected max_tokens={data['output_tokens']}"
            )

        backend = config["backend"]
        validate_sampling(
            backend["name"],
            backend.get("mode", "chat"),
            sampling,
        )

        if model["id"] in model_hashes:
            raise ValueError("Duplicate model")
        model_hashes[model["id"]] = digest(config_path)

    preset_hashes = {}
    allowed = {"baseline", "sillytavern", "megumin-v9", "arfyjb"}

    for preset in data["presets"]:
        kind = preset["adapter"]
        if kind not in allowed:
            raise ValueError(f"Unknown adapter: {kind}")
        if preset["id"] in preset_hashes:
            raise ValueError("Duplicate preset")

        inputs = {}
        if kind == "sillytavern":
            inputs["source"] = preset["source"]
            if preset.get("backend_compat") not in {
                "alternating-chat-v1",
                "alternating-chat-v2",
            }:
                raise ValueError("Missing SillyTavern compatibility mode")
        elif kind == "megumin-v9":
            inputs["config"] = preset["config"]
            directory = Path(preset["engine_directory"])
            inputs["engine"] = str(directory / "engine.json")
            inputs["cot"] = str(directory / "cot.json")
            profile = json.loads(Path(preset["config"]).read_text())
            inputs["source"] = profile["source"]["preset"]
        elif kind == "arfyjb":
            inputs["source"] = preset["source"]

        preset_hashes[preset["id"]] = {
            name: digest(path) for name, path in inputs.items()
        }

    count = len(case_hashes) * len(model_hashes) * len(preset_hashes)

    return {
        "schema_version": 1,
        "output_tokens": data["output_tokens"],
        "scenario_manifest_sha256": digest(manifest),
        "scenarios": case_hashes,
        "models": model_hashes,
        "presets": preset_hashes,
        "planned_generations": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        parser.error(
            "Execution is not enabled yet. Use --dry-run."
        )

    data = load_matrix(args.matrix)
    plan = validate_matrix(data)

    print("=== RPBENCH EXPERIMENT MATRIX ===")
    print("Output tokens:", plan["output_tokens"])
    print("Models:", len(plan["models"]))
    print("Presets:", len(plan["presets"]))
    print("Scenarios:", len(plan["scenarios"]))
    print("Planned generations:", plan["planned_generations"])

    print("\nModels:")
    for name in plan["models"]:
        print(" ", name)

    print("\nPresets:")
    for name in plan["presets"]:
        print(" ", name)

    print("\nDry-run validation: PASS")


if __name__ == "__main__":
    main()
