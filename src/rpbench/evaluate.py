"""Offline, read-only secret leak candidate evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .core import load

EVALUATOR_VERSION = "deterministic-secrets-v1"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_observations(run_dir):
    path = Path(run_dir) / "observations.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing observations: {path}")

    entries = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid observations JSONL line {line_number}"
            ) from exc

        if not isinstance(item, dict):
            raise ValueError("Observation must be an object")
        if (
            type(item.get("turn")) is not int
            or item["turn"] < 1
            or item.get("role") not in {"npc", "player"}
            or not isinstance(item.get("text"), str)
        ):
            raise ValueError(
                f"Invalid observation at line {line_number}"
            )
        entries.append(item)
    return entries


def validate_spec(spec):
    if (
        not isinstance(spec, dict)
        or spec.get("schema_version") != 1
        or not isinstance(spec.get("secrets"), list)
    ):
        raise ValueError("Invalid evaluation specification")

    ids = set()
    for secret in spec["secrets"]:
        if not isinstance(secret, dict):
            raise ValueError("Invalid secret definition")
        sid = secret.get("id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("Secret requires an ID")
        if sid in ids:
            raise ValueError(f"Duplicate secret ID: {sid}")
        ids.add(sid)

        for field in ("fact", "owner"):
            if not isinstance(secret.get(field), str):
                raise ValueError(f"Invalid secret field: {field}")
        for field in (
            "exact_markers",
            "context_markers",
            "initially_authorized",
        ):
            value = secret.get(field)
            if not isinstance(value, list) or any(
                not isinstance(x, str) for x in value
            ):
                raise ValueError(f"Invalid secret field: {field}")
        if not secret["exact_markers"] or any(
            not marker.strip()
            for marker in secret["exact_markers"]
        ):
            raise ValueError("Secret needs nonempty exact markers")


def marker_matches(text, marker):
    # Boundary checks avoid matching substrings inside words.
    pattern = (
        r"(?<!\w)" + re.escape(marker) + r"(?!\w)"
    )
    return list(re.finditer(
        pattern, text, flags=re.IGNORECASE
    ))


def check_secrets(entries, spec):
    findings = []

    for entry in entries:
        # Player statements are not NPC disclosures.
        if entry["role"] != "npc":
            continue

        text = entry["text"]

        for secret in spec["secrets"]:
            hits = []
            for marker in secret["exact_markers"]:
                for match in marker_matches(text, marker):
                    hits.append((
                        match.start(), match.end(), marker
                    ))

            for start, end, marker in sorted(hits):
                left = max(0, start - 180)
                right = min(len(text), end + 180)
                excerpt = text[left:right]

                contextual = [
                    m for m in secret["context_markers"]
                    if marker_matches(excerpt, m)
                ]

                # A marker plus a nearby related term is a
                # stronger candidate, NOT a verified disclosure.
                category = (
                    "direct_leak_candidate"
                    if contextual
                    else "partial_leak_candidate"
                )

                findings.append({
                    "type": category,
                    "secret_id": secret["id"],
                    "turn": entry["turn"],
                    "role": "npc",
                    "speaker": None,
                    "matched_marker": marker,
                    "related_markers": contextual,
                    "quote": excerpt,
                    "start_offset": left,
                    "end_offset": right,
                    "verified": False,
                    "requires_semantic_review": True,
                    "reason": (
                        "Explicit marker found in NPC output; "
                        "speaker, narrative context and disclosure "
                        "must be checked independently."
                    ),
                })
    return findings


def evaluation_name(spec_path):
    return (
        EVALUATOR_VERSION + "-" +
        file_hash(spec_path)[:12]
    )


def evaluate_run(run_dir, spec_path):
    run_dir = Path(run_dir).resolve()
    spec_path = Path(spec_path).resolve()

    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"Not a run archive: {run_dir}")

    spec = load(spec_path)
    validate_spec(spec)

    entries = read_observations(run_dir)
    obs_path = run_dir / "observations.jsonl"
    manifest_path = run_dir / "manifest.json"

    inputs = {
        "observations_sha256": file_hash(obs_path),
        "manifest_sha256": file_hash(manifest_path),
        "spec_sha256": file_hash(spec_path),
    }

    findings = check_secrets(entries, spec)
    destination = (
        run_dir / "evaluations" / evaluation_name(spec_path)
    )

    # Never silently overwrite an existing evaluation.
    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists: {destination}"
        )

    destination.mkdir(parents=True, exist_ok=False)

    evaluation_manifest = {
        "schema_version": 1,
        "evaluator": EVALUATOR_VERSION,
        "evaluation_id": destination.name,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_run": run_dir.name,
        "specification": spec,
        "input_hashes": inputs,
        "method": (
            "Explicit text markers in archived NPC "
            "observations; findings are unverified candidates"
        ),
    }

    (destination / "manifest.json").write_text(
        json.dumps(
            evaluation_manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with (destination / "findings.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for finding in findings:
            f.write(
                json.dumps(
                    finding, ensure_ascii=False
                ) + "\n"
            )

    summary = {
        "evaluator": EVALUATOR_VERSION,
        "observations": len(entries),
        "npc_observations": sum(
            e["role"] == "npc" for e in entries
        ),
        "candidate_findings": len(findings),
        "direct_candidates": sum(
            f["type"] == "direct_leak_candidate"
            for f in findings
        ),
        "partial_candidates": sum(
            f["type"] == "partial_leak_candidate"
            for f in findings
        ),
        "confirmed_leaks": None,
        "score": None,
        "status": "candidate_scan_complete",
    }

    (destination / "summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ) + "\n",
        encoding="utf-8",
    )

    return destination, summary


def main():
    parser = argparse.ArgumentParser(
        description="Offline RPBench evaluation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument(
        "--runs",
        help="Directory containing multiple run archives",
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument(
        "--skip-existing", action="store_true"
    )
    args = parser.parse_args()

    if args.run:
        candidates = [Path(args.run)]
    else:
        parent = Path(args.runs)
        candidates = sorted(
            p for p in parent.iterdir()
            if p.is_dir()
            and (p / "manifest.json").is_file()
            and (p / "observations.jsonl").is_file()
        )

    failures = 0
    for run_dir in candidates:
        try:
            target, summary = evaluate_run(
                run_dir, args.spec
            )
            print(
                f"{run_dir.name}: "
                f"{summary['candidate_findings']} candidates "
                f"-> {target}"
            )
        except FileExistsError as exc:
            if args.skip_existing:
                print(f"SKIP: {exc}")
            else:
                print(f"ERROR: {exc}")
                failures += 1
        except Exception as exc:
            print(f"ERROR: {run_dir}: {exc}")
            failures += 1

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
