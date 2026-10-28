"""Resumable evaluation of a fixed-history refusal batch."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .core import dump
from .refusal_judge import evaluate_probe
from .rp_dimension_judge import create_backend
from .sampling import validate_sampling


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_probes(directory):
    directory = Path(directory)
    found = {}

    for filename in sorted(directory.glob("*/result.json")):
        data = filename.read_bytes()
        probe = json.loads(data)
        scenario = probe["scenario_id"]

        if scenario in found:
            raise ValueError(
                f"Multiple probe attempts for {scenario}; "
                "select a unique completed attempt first"
            )

        if probe.get("status") != "complete":
            raise ValueError(f"Incomplete probe: {scenario}")

        if "model_identity" not in probe:
            raise ValueError(f"Missing model identity: {scenario}")

        found[scenario] = {
            "probe": probe,
            "digest": digest(data),
            "file": filename,
        }

    if not found:
        raise ValueError("No completed probes found")

    return found


def read_attempts(directory, probes, config_digest):
    attempts = {}

    for filename in sorted(Path(directory).glob("*/*.json")):
        record = json.loads(filename.read_text(encoding="utf-8"))
        scenario = record.get("scenario_id")

        if scenario not in probes:
            raise ValueError(f"Unexpected archived judgment: {filename}")

        if record.get("judge_config_sha256") != config_digest:
            raise ValueError(f"Judge configuration changed: {filename}")

        if record.get("probe_sha256") != probes[scenario]["digest"]:
            raise ValueError(f"Probe content changed: {filename}")

        attempts.setdefault(scenario, []).append(record)

    return attempts


def pending_probes(probes, attempts, retry_failed=False):
    pending = []

    for scenario in sorted(probes):
        rows = attempts.get(scenario, [])
        if any(row.get("valid") is True for row in rows):
            continue
        if rows and not retry_failed:
            continue
        pending.append(scenario)

    return pending


def run_batch(probe_dir, judge_config, output, retry_failed=False):
    probes = load_probes(probe_dir)

    config_bytes = Path(judge_config).read_bytes()
    config_digest = digest(config_bytes)
    config = json.loads(config_bytes)

    backend = create_backend(config)
    sampling = config["sampling"]
    validate_sampling(backend.name, backend.mode, sampling)

    output = Path(output)
    attempts = read_attempts(output, probes, config_digest)
    pending = pending_probes(probes, attempts, retry_failed)

    print(
        f"Probes={len(probes)} "
        f"archived={sum(map(len, attempts.values()))} "
        f"pending={len(pending)}",
        flush=True,
    )

    failures = 0

    for index, scenario in enumerate(pending, 1):
        item = probes[scenario]

        print(
            f"{index:02}/{len(pending):02} | {scenario}",
            flush=True,
        )

        result = evaluate_probe(
            item["probe"], backend, sampling
        )

        result.update({
            "scenario_id": scenario,
            "category": item["probe"]["category"],
            "model_identity": item["probe"]["model_identity"],
            "probe_sha256": item["digest"],
            "judge_config_sha256": config_digest,
            "judge_model": backend.model,
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        })

        target_dir = output / scenario
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        target = target_dir / f"{timestamp}.json"
        dump(target, result)

        if not result["valid"]:
            failures += 1

        verdict = result.get("verdict") or {}
        print(
            f"  valid={result['valid']} "
            f"met={result.get('expectation_met')} "
            f"refusal={verdict.get('model_refusal')} "
            f"unsafe={verdict.get('unsafe_actionable')} "
            f"sexual={verdict.get('unsafe_sexual_content')} "
            f"time={result['elapsed_seconds']}s",
            flush=True,
        )

    print(
        f"Finished: {len(pending) - failures} valid, "
        f"{failures} invalid",
        flush=True,
    )
    print("Archive:", output, flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", required=True)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    failures = run_batch(
        args.probes,
        args.judge_config,
        args.out,
        retry_failed=args.retry_failed,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
