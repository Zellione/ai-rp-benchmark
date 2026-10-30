"""Resumable, fingerprinted Megumin V9 refusal batches."""

import argparse
import hashlib
import json
from pathlib import Path

from .core import dump
from .deployment_preflight import get_json, verify
from .refusal_batch import load_plan
from .refusal_preset_batch import safe_name
from .refusal_probe import run_probe


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return digest(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8"))


def experiment_identity(config, config_sha, megumin_config, engine_dir):
    config_file = Path(megumin_config)
    engine_dir = Path(engine_dir)

    profile_bytes = config_file.read_bytes()
    profile = json.loads(profile_bytes)
    source = Path(profile["source"]["preset"])

    source_sha = digest(source.read_bytes())
    if source_sha != profile["source"]["preset_sha256"]:
        raise ValueError("Megumin source fingerprint mismatch")

    engine_bytes = (engine_dir / "engine.json").read_bytes()
    cot_bytes = (engine_dir / "cot.json").read_bytes()

    engine = json.loads(engine_bytes)
    cot = json.loads(cot_bytes)

    if engine["id"] != profile["engine"]:
        raise ValueError("Megumin engine mismatch")
    if cot["id"] != profile["cot"]:
        raise ValueError("Megumin CoT mismatch")

    return {
        "schema_version": 1,
        "adapter": "megumin-v9",
        "model_identity": config["model_identity"],
        "config_sha256": config_sha,
        "preset_id": "megumin-v9-mirage",
        "preset_sha256": source_sha,
        "megumin_config_sha256": digest(profile_bytes),
        "engine_sha256": digest(engine_bytes),
        "cot_sha256": digest(cot_bytes),
        "character_id": profile["character_id"],
        "backend_compat": "megumin-prefinal-system+alternating-chat",
        "prefill_applied": False,
    }


def experiment_directory(root, identity):
    model = identity["model_identity"]
    stem = safe_name(
        f"{model['id']}__{model['variant']}__"
        f"{model['quantization']}__megumin-v9-mirage"
    )
    fingerprint = canonical_digest(identity)[:16]
    return Path(root) / f"{stem}__{fingerprint}"


def validate_archive(directory, identity, plan):
    manifest = directory / "experiment.json"

    if directory.exists():
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved != identity:
                raise ValueError("Experiment identity mismatch")
        elif any(directory.iterdir()):
            raise ValueError("Existing nonempty experiment lacks manifest")

    expected = {
        item["id"]: item["scenario_sha256"] for item in plan
    }
    archived = {}

    for path in sorted(directory.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row.get("scenario_id")

        if scenario not in expected:
            raise ValueError(f"Unexpected archived scenario: {path}")
        if row.get("scenario_sha256") != expected[scenario]:
            raise ValueError(f"Scenario hash mismatch: {path}")
        if row.get("config_sha256") != identity["config_sha256"]:
            raise ValueError(f"Model config mismatch: {path}")
        if row.get("model_identity") != identity["model_identity"]:
            raise ValueError(f"Model mismatch: {path}")

        preset = row.get("preset") or {}
        fields = {
            "id": "preset_id",
            "source_sha256": "preset_sha256",
            "config_sha256": "megumin_config_sha256",
            "engine_sha256": "engine_sha256",
            "cot_sha256": "cot_sha256",
            "character_id": "character_id",
            "prefill_applied": "prefill_applied",
        }
        for row_field, identity_field in fields.items():
            if preset.get(row_field) != identity[identity_field]:
                raise ValueError(
                    f"Archived {row_field} mismatch: {path}"
                )

        compat = preset.get("backend_compat") or {}
        if compat.get("strategy") not in {
            "alternating-chat-v1",
            "alternating-chat-v2",
        }:
            raise ValueError(f"Unknown lowering: {path}")

        archived.setdefault(scenario, []).append(row)

    return archived


def pending_cases(plan, archived, retry_failed):
    pending = []
    for item in plan:
        attempts = archived.get(item["id"], [])
        if any(r.get("status") == "complete" for r in attempts):
            continue
        if attempts and not retry_failed:
            continue
        pending.append(item)
    return pending


def run_batch(
    manifest,
    config_file,
    megumin_config,
    engine_dir,
    output,
    *,
    retry_failed=False,
):
    plan, model, config_sha = load_plan(manifest, config_file)
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))

    identity = experiment_identity(
        config, config_sha, megumin_config, engine_dir
    )

    directory = experiment_directory(output, identity)
    archived = validate_archive(directory, identity, plan)
    pending = pending_cases(plan, archived, retry_failed)

    print("Experiment:", directory.name)
    print("Planned:", len(plan))
    print("Already archived:", len(archived))
    print("Pending:", len(pending))

    if pending:
        backend = config["backend"]
        if backend["name"] != "mock":
            base = backend["base_url"].rstrip("/")
            models = get_json(base + "/v1/models")
            props = (
                get_json(base + "/props")
                if backend["name"] == "llamacpp"
                else None
            )
            print(
                "Preflight:",
                json.dumps(verify(config, models, props)),
            )

    directory.mkdir(parents=True, exist_ok=True)
    experiment_file = directory / "experiment.json"

    if not experiment_file.exists():
        dump(experiment_file, identity)

    failures = 0

    for index, item in enumerate(pending, 1):
        print(
            f"{index:02d}/{len(pending):02d} | {item['id']}",
            flush=True,
        )

        row, _ = run_probe(
            item["file"],
            config_file,
            directory,
            megumin_config=megumin_config,
            megumin_engine_directory=engine_dir,
        )

        if row["status"] != "complete":
            failures += 1
            print("Unsuccessful:", row["status"], flush=True)

    print(
        f"Finished: {len(pending) - failures} complete, "
        f"{failures} unsuccessful"
    )
    print("Archive:", directory)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--megumin-config", required=True)
    parser.add_argument("--engine-directory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    failures = run_batch(
        args.manifest,
        args.config,
        args.megumin_config,
        args.engine_directory,
        args.out,
        retry_failed=args.retry_failed,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
