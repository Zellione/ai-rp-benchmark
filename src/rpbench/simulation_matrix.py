"""Single-model, bilingual simulation experiment orchestration."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from .simulation import run_simulation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def identifier(value):
    if (
        not isinstance(value, str)
        or not value
        or any(c not in "abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
               for c in value)
    ):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def resolve(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected nonempty file path")
    path = (base / value).resolve()
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    return path


def plan(matrix_path):
    matrix_path = Path(matrix_path).resolve()
    matrix = load_json(matrix_path)
    base = matrix_path.parent

    if matrix.get("schema_version") != 1:
        raise ValueError("Unsupported simulation matrix schema")

    model = identifier(matrix["model"]["id"])
    turns = matrix["turns"]

    if type(turns) is not int or not 1 <= turns <= 400:
        raise ValueError("turns must be 1..400")

    scenarios = matrix["scenarios"]
    presets = matrix["presets"]

    if not scenarios or not presets:
        raise ValueError("Empty scenario or preset matrix")

    if {entry["language"] for entry in scenarios} != {"de", "en"}:
        raise ValueError("Both de and en scenarios are required")

    scenario_keys = []
    for entry in scenarios:
        language = entry["language"]
        scenario_id = identifier(entry["id"])
        scenario_keys.append((language, scenario_id))

    if len(scenario_keys) != len(set(scenario_keys)):
        raise ValueError("Duplicate language/scenario combination")

    preset_ids = [identifier(p["id"]) for p in presets]
    if len(preset_ids) != len(set(preset_ids)):
        raise ValueError("Duplicate preset ID")

    jobs = []

    for scenario, preset in itertools.product(scenarios, presets):
        language = scenario["language"]
        scenario_id = identifier(scenario["id"])
        preset_id = identifier(preset["id"])

        scenario_path = resolve(base, scenario["scenario"])
        player_path = resolve(base, scenario["player"])
        config_path = resolve(base, preset["config"])

        config = load_json(config_path)
        backend = config["backend"]

        if backend.get("model") != matrix["model"]["backend_model"]:
            raise ValueError(
                f"Wrong model in preset config: {config_path}"
            )

        source_digests = {
            "scenario": digest(scenario_path),
            "player": digest(player_path),
            "config": digest(config_path),
        }

        identity = {
            "model": model,
            "backend_model": matrix["model"]["backend_model"],
            "language": language,
            "scenario": scenario_id,
            "preset": preset_id,
            "turns": turns,
            "sources": source_digests,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        jobs.append({
            "key": (
                f"{model}__{language}__{scenario_id}"
                f"__{preset_id}__{fingerprint}"
            ),
            "identity": identity,
            "scenario_path": scenario_path,
            "player_path": player_path,
            "config_path": config_path,
        })

    return jobs


def execute(jobs, output, dry_run=False, limit=None):
    output = Path(output)

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    selected = jobs if limit is None else jobs[:limit]

    completed = 0
    skipped = 0

    for index, job in enumerate(selected, 1):
        root = output / job["key"]
        state_path = root / "job.json"

        print(
            f"[{index}/{len(selected)}] {job['key']}",
            flush=True,
        )

        if state_path.is_file():
            state = load_json(state_path)
            if state.get("identity") != job["identity"]:
                raise ValueError(f"Identity mismatch: {state_path}")

            if state["status"] == "complete":
                run_dir = Path(state["run_dir"])
                manifest_path = run_dir / "manifest.json"

                if (
                    not manifest_path.is_file()
                    or load_json(manifest_path).get("status") != "complete"
                ):
                    raise ValueError(f"Broken completed archive: {run_dir}")

                print("  archived; skipping", flush=True)
                skipped += 1
                continue

            raise ValueError(
                f"Previous unfinished job: {state_path}. "
                "Inspect or resume its simulation archive before retrying."
            )

        if dry_run:
            print("  dry-run: no files written", flush=True)
            continue

        # A job directory is unique to its immutable experiment inputs.
        # Existing unfinished runs must never be silently overwritten.
        root.mkdir(parents=True, exist_ok=False)

        state = {
            "schema_version": 1,
            "status": "running",
            "identity": job["identity"],
        }
        save_json(state_path, state)

        try:
            run_dir = run_simulation(
                scenario_path=job["scenario_path"],
                player_path=job["player_path"],
                config_path=job["config_path"],
                output=root / "runs",
                turns=job["identity"]["turns"],
            )
        except Exception as error:
            state["status"] = "failed"
            state["error"] = repr(error)
            save_json(state_path, state)
            raise

        state["run_dir"] = str(Path(run_dir).resolve())
        state["status"] = "complete"
        save_json(state_path, state)

        completed += 1
        print("  complete:", run_dir, flush=True)

    print(
        f"Planned={len(jobs)} Selected={len(selected)} "
        f"Completed={completed} Skipped={skipped}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    jobs = plan(args.matrix)

    execute(
        jobs,
        args.out,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
