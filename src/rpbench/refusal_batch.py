"""Resumable fixed-history refusal benchmark batch."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .refusal_probe import (
    run_probe,
    validate_model_identity,
    validate_scenario,
)
from .deployment_preflight import get_json, verify


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def deployment_name(identity):
    name = (
        f"{identity['id']}__"
        f"{identity['variant']}__"
        f"{identity['quantization']}"
    )
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def load_plan(manifest_file, config_file):
    manifest_file = Path(manifest_file)
    config_file = Path(config_file)

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)

    identity = validate_model_identity(config["model_identity"])

    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("cases"), list)
        or not manifest["cases"]
    ):
        raise ValueError("Invalid refusal manifest")

    plan = []
    seen = set()

    for item in manifest["cases"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid manifest entry")

        filename = item.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".json")
        ):
            raise ValueError("Invalid scenario filename")

        scenario_file = manifest_file.parent / filename
        scenario_bytes = scenario_file.read_bytes()
        scenario = json.loads(scenario_bytes)

        validate_scenario(scenario)

        identifier = scenario["id"]
        if identifier in seen:
            raise ValueError("Duplicate scenario ID")
        seen.add(identifier)

        for field, expected in (
            ("id", identifier),
            ("category", scenario["category"]),
            ("expected_behavior", scenario["expected_behavior"]),
            ("probe_turn", scenario["probe_turn"]),
        ):
            if item.get(field) != expected:
                raise ValueError(f"Manifest mismatch: {field}")

        plan.append({
            "id": identifier,
            "file": scenario_file,
            "scenario_sha256": sha256(scenario_bytes),
        })

    return plan, identity, sha256(config_bytes)


def archived_attempts(directory, identity, config_digest):
    records = {}

    if not directory.exists():
        return records

    for filename in sorted(directory.glob("*/result.json")):
        row = json.loads(filename.read_text(encoding="utf-8"))

        if row.get("model_identity") != identity:
            raise ValueError(f"Model identity mismatch: {filename}")

        if row.get("config_sha256") != config_digest:
            raise ValueError(f"Config mismatch: {filename}")

        identifier = row.get("scenario_id")
        records.setdefault(identifier, []).append(row)

    return records


def pending_plan(plan, archived, retry_failed=False):
    planned = {item["id"] for item in plan}

    unexpected = set(archived) - planned
    if unexpected:
        raise ValueError(f"Unexpected archived scenarios: {unexpected}")

    pending = []
    for item in plan:
        attempts = archived.get(item["id"], [])

        if any(
            row.get("scenario_sha256") != item["scenario_sha256"]
            for row in attempts
        ):
            raise ValueError(f"Scenario changed: {item['id']}")

        if any(row.get("status") == "complete" for row in attempts):
            continue

        if attempts and not retry_failed:
            continue

        pending.append(item)

    return pending


def run_batch(manifest_file, config_file, output, retry_failed=False):
    plan, identity, config_digest = load_plan(
        manifest_file, config_file
    )

    directory = Path(output) / deployment_name(identity)
    archived = archived_attempts(directory, identity, config_digest)
    pending = pending_plan(plan, archived, retry_failed)

    completed = sum(
        any(row.get("status") == "complete" for row in rows)
        for rows in archived.values()
    )

    print(
        f"Deployment: {identity['id']} "
        f"[{identity['quantization']}]",
        flush=True,
    )
    print(
        f"Planned={len(plan)} "
        f"completed={completed} "
        f"pending={len(pending)}",
        flush=True,
    )

    directory.mkdir(parents=True, exist_ok=True)

    if pending:
        config = json.loads(
            Path(config_file).read_text(encoding="utf-8")
        )
        backend = config["backend"]

        if backend["name"] != "mock":
            base = backend["base_url"].rstrip("/")
            models = get_json(base + "/v1/models")
            props = (
                get_json(base + "/props")
                if backend["name"] == "llamacpp"
                else None
            )

            verification = verify(config, models, props)
            print(
                "Preflight:",
                json.dumps(verification),
                flush=True,
            )

    failures = 0

    for index, item in enumerate(pending, 1):
        print(
            f"{index:02}/{len(pending):02} | {item['id']}",
            flush=True,
        )

        record, _ = run_probe(
            item["file"], config_file, directory
        )
        if record["status"] != "complete":
            failures += 1
            print(
                f"  ERROR: {record.get('error', record['status'])}",
                flush=True,
            )

    print(
        f"Batch finished: {len(pending) - failures} completed, "
        f"{failures} unsuccessful",
        flush=True,
    )
    print("Archive:", directory, flush=True)

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    failures = run_batch(
        args.manifest,
        args.config,
        args.out,
        retry_failed=args.retry_failed,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
