"""Cross-deployment descriptive report for fixed-history refusal probes."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .core import dump
from .refusal_compare import summarize


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refusal_judgments(root, deployment):
    recovered = root / "judgments-recovered-v1" / deployment

    if recovered.is_dir():
        files = sorted(recovered.glob("*/verdict.json"))
        return {
            row["scenario_id"]: row
            for path in files
            if (row := read_json(path))
        }

    source = root / "judgments" / deployment
    result = {}

    by_scenario = {}
    for path in sorted(source.glob("*/*.json")):
        row = read_json(path)
        by_scenario.setdefault(row["scenario_id"], []).append(row)

    for scenario, attempts in by_scenario.items():
        valid = [row for row in attempts if row.get("valid") is True]
        if len(valid) > 1:
            raise ValueError(f"Multiple valid judgments: {scenario}")
        if valid:
            result[scenario] = valid[0]

    return result


def fulfillment_judgments(root, deployment):
    directory = root / "fulfillment" / deployment
    result = {}

    for path in sorted(directory.glob("*/verdict.json")):
        row = read_json(path)
        scenario = row["scenario_id"]

        if scenario in result:
            raise ValueError(f"Duplicate fulfillment: {scenario}")
        result[scenario] = row

    return result


def summarize_deployment(root, deployment, expected_scenarios):
    probes = root / "batches" / deployment
    refusals = refusal_judgments(root, deployment)
    fulfillment = fulfillment_judgments(root, deployment)

    # Reuse the existing reporter for generation reliability and
    # its model-identity consistency checks.
    # Materialize only valid selected judgments in a temporary-free
    # logical scan below, so deployments with incomplete judgments work.
    attempts = {}
    identity = None
    for path in sorted(probes.glob("*/result.json")):
        row = read_json(path)
        scenario = row["scenario_id"]
        attempts.setdefault(scenario, []).append(row)

        if identity is None:
            identity = row["model_identity"]
        elif identity != row["model_identity"]:
            raise ValueError(f"Mixed identities: {deployment}")

    if not attempts:
        raise ValueError(f"Empty deployment: {deployment}")

    if set(attempts) != expected_scenarios:
        raise ValueError(f"Scenario coverage mismatch: {deployment}")

    if set(refusals) - expected_scenarios:
        raise ValueError(f"Unknown refusal judgment: {deployment}")
    if set(fulfillment) - expected_scenarios:
        raise ValueError(f"Unknown fulfillment judgment: {deployment}")

    cases = []
    for scenario in sorted(expected_scenarios):
        rows = attempts[scenario]
        completed = [r for r in rows if r["status"] == "complete"]

        if len(completed) > 1:
            raise ValueError(f"Multiple completed probes: {scenario}")

        refusal = refusals.get(scenario)
        full = fulfillment.get(scenario)

        if refusal and refusal.get("valid") is not True:
            raise ValueError(f"Invalid selected refusal: {scenario}")

        if refusal and not completed:
            raise ValueError(f"Refusal without generation: {scenario}")
        if full and full.get("valid") is True and not completed:
            raise ValueError(f"Fulfillment without generation: {scenario}")

        for judgment in (refusal, full):
            if judgment and judgment.get("model_identity") != identity:
                raise ValueError(f"Judgment identity mismatch: {scenario}")

        rv = refusal["verdict"] if refusal else None
        fv = full["verdict"] if full and full.get("valid") else None

        cases.append({
            "scenario_id": scenario,
            "category": rows[0]["category"],
            "generated": bool(completed),
            "attempts": len(rows),
            "statuses": [r["status"] for r in rows],
            "refusal": rv["model_refusal"] if rv else None,
            "refusal_style": rv["refusal_style"] if rv else None,
            "rp_continuity": rv["roleplay_continuity"] if rv else None,
            "actionable_harm": rv["unsafe_actionable"] if rv else None,
            "sexual_content": rv["unsafe_sexual_content"] if rv else None,
            "fulfillment": fv["fulfillment"] if fv else None,
            "response_mode": fv["response_mode"] if fv else None,
        })

    judged_r = [c for c in cases if c["refusal"] is not None]
    judged_f = [c for c in cases if c["fulfillment"] is not None]

    return {
        "deployment": deployment,
        "model_identity": identity,
        "backend": completed_backend(attempts),
        "scenarios": len(cases),
        "generated": sum(c["generated"] for c in cases),
        "generation_attempts": sum(c["attempts"] for c in cases),
        "generation_statuses": dict(Counter(
            s for c in cases for s in c["statuses"]
        )),
        "refusal_evaluated": len(judged_r),
        "fulfillment_evaluated": len(judged_f),
        "model_refusals": sum(c["refusal"] for c in judged_r),
        "refusal_styles": dict(Counter(
            c["refusal_style"] for c in judged_r
        )),
        "rp_continuations": sum(c["rp_continuity"] for c in judged_r),
        "actionable_harm_flags": sum(
            c["actionable_harm"] for c in judged_r
        ),
        "sexual_content_flags": sum(
            c["sexual_content"] for c in judged_r
        ),
        "fulfillment_levels": dict(Counter(
            c["fulfillment"] for c in judged_f
        )),
        "cases": cases,
    }


def completed_backend(attempts):
    backends = {
        json.dumps(row["backend"], sort_keys=True)
        for rows in attempts.values()
        for row in rows
    }
    if len(backends) != 1:
        raise ValueError("Mixed backend configurations")
    return json.loads(next(iter(backends)))


def build_report(root, manifest):
    root = Path(root)
    manifest = read_json(Path(manifest))
    expected = {entry["id"] for entry in manifest["cases"]}

    if len(expected) != len(manifest["cases"]):
        raise ValueError("Duplicate scenario in manifest")

    directory = root / "batches"
    deployments = sorted(
        p.name for p in directory.iterdir()
        if p.is_dir() and list(p.glob("*/result.json"))
    )

    if not deployments:
        raise ValueError("No deployments found")

    results = [
        summarize_deployment(root, name, expected)
        for name in deployments
    ]

    return {
        "schema_version": 1,
        "benchmark": "refusal-pilot-v1",
        "scenario_ids": sorted(expected),
        "deployment_count": len(results),
        "deployments": results,
    }


def write_table(path, report):
    columns = [
        "deployment", "model_id", "variant", "quantization",
        "generated", "scenarios", "attempts",
        "refusal_evaluated", "model_refusals",
        "fulfillment_evaluated", "full", "substantial",
        "partial", "none", "rp_continuations",
        "actionable_harm_flags", "sexual_content_flags",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in report["deployments"]:
            identity = row["model_identity"]
            levels = row["fulfillment_levels"]
            writer.writerow({
                "deployment": row["deployment"],
                "model_id": identity["id"],
                "variant": identity["variant"],
                "quantization": identity["quantization"],
                "generated": row["generated"],
                "scenarios": row["scenarios"],
                "attempts": row["generation_attempts"],
                "refusal_evaluated": row["refusal_evaluated"],
                "model_refusals": row["model_refusals"],
                "fulfillment_evaluated": row["fulfillment_evaluated"],
                **{key: levels.get(key, 0) for key in
                   ("full", "substantial", "partial", "none")},
                "rp_continuations": row["rp_continuations"],
                "actionable_harm_flags": row["actionable_harm_flags"],
                "sexual_content_flags": row["sexual_content_flags"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    destination = Path(args.out)
    table = destination.with_suffix(".tsv")
    if destination.exists() or table.exists():
        parser.error("Output already exists")

    report = build_report(args.root, args.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump(destination, report)
    write_table(table, report)

    print("Deployments:", report["deployment_count"])
    print("Saved:", destination)
    print("Table:", table)


if __name__ == "__main__":
    main()
