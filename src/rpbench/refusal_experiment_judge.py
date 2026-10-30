"""Unified, resumable refusal and fulfillment judge orchestration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .refusal_experiment import load_matrix, validate_matrix
from .refusal_experiment_run import (
    build_identity,
    experiment_dir,
    validate_archive,
)
from .refusal_batch import load_plan


def discover_experiments(matrix_path, archive, model_filter=None):
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)

    results = []

    for model in matrix["models"]:
        if model_filter and model["id"] != model_filter:
            continue

        config_path = Path(model["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))

        plan, _, config_digest = load_plan(
            matrix["scenario_manifest"], config_path
        )

        for preset in matrix["presets"]:
            identity = build_identity(
                model, preset, config, config_digest
            )
            directory = experiment_dir(archive, identity)

            if not (directory / "experiment.json").exists():
                raise ValueError(
                    f"Missing experiment archive: {directory}"
                )

            archived = validate_archive(directory, identity, plan)

            completed = {
                scenario
                for scenario, attempts in archived.items()
                if any(
                    row.get("status") == "complete"
                    for row in attempts
                )
            }

            results.append({
                "model": model["id"],
                "preset": preset["id"],
                "directory": directory,
                "completed": len(completed),
                "planned": len(plan),
            })

    if not results:
        raise ValueError("No matching experiments")

    return results


def judge_commands(experiment, judge_config, output, retry_failed):
    directory = experiment["directory"]
    name = directory.name

    refusal = [
        sys.executable,
        "-m", "rpbench.refusal_judge_batch",
        "--probes", str(directory),
        "--judge-config", str(judge_config),
        "--out", str(output / "refusal" / name),
        "--allow-incomplete",
    ]

    fulfillment = [
        sys.executable,
        "-m", "rpbench.refusal_fulfillment",
        "--probes", str(directory),
        "--judge-config", str(judge_config),
        "--out", str(output / "fulfillment" / name),
    ]

    if retry_failed:
        refusal.append("--retry-failed")
        fulfillment.append("--retry-failed")

    return [
        ("refusal", refusal),
        ("fulfillment", fulfillment),
    ]


def execute(
    matrix,
    archive,
    judge_config,
    output,
    *,
    model=None,
    retry_failed=False,
    dry_run=False,
):
    experiments = discover_experiments(
        matrix, archive, model_filter=model
    )

    judge_config = Path(judge_config)
    if not judge_config.is_file():
        raise ValueError(f"Missing judge config: {judge_config}")

    output = Path(output)
    failures = []

    print("=== UNIFIED JUDGE PIPELINE ===")
    print("Experiments:", len(experiments))
    print("Judge tasks:", len(experiments) * 2)

    for index, experiment in enumerate(experiments, 1):
        label = f"{experiment['model']} / {experiment['preset']}"

        print(
            f"\n[{index}/{len(experiments)}] {label} "
            f"complete={experiment['completed']}/{experiment['planned']}",
            flush=True,
        )

        for kind, command in judge_commands(
            experiment, judge_config, output, retry_failed
        ):
            print(f"  Judge: {kind}", flush=True)

            if dry_run:
                print("  Dry run: no judge request")
                continue

            result = subprocess.run(command, check=False)

            if result.returncode != 0:
                failures.append({
                    "model": experiment["model"],
                    "preset": experiment["preset"],
                    "judge": kind,
                    "returncode": result.returncode,
                })
                print(
                    f"  WARNING: {kind} exited with "
                    f"{result.returncode}; continuing",
                    flush=True,
                )

    print("\n=== SUMMARY ===")
    print("Experiments:", len(experiments))
    print("Failed judge invocations:", len(failures))

    for failure in failures:
        print(
            f"  {failure['model']} / {failure['preset']} / "
            f"{failure['judge']}: exit={failure['returncode']}"
        )

    if dry_run:
        print("No requests or archive writes performed")

    return len(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    failures = execute(
        args.matrix,
        args.archive,
        args.judge_config,
        args.out,
        model=args.model,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
