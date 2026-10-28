"""Deterministic reporting for independent RP quality judgments."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from .rp_quality_judge import DIMENSIONS


def read_jsonl(filename):
    records = []
    for number, line in enumerate(
        Path(filename).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at line {number}"
            ) from exc
    return records


def build_report(dataset_file, results_file):
    dataset_bytes = Path(dataset_file).read_bytes()
    dataset = json.loads(dataset_bytes)
    records = read_jsonl(results_file)

    if not isinstance(dataset.get("cases"), list):
        raise ValueError("Invalid candidate dataset")
    if not records:
        raise ValueError("No judgment records")

    ids = [case["id"] for case in dataset["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")

    digest = hashlib.sha256(dataset_bytes).hexdigest()

    grouped = defaultdict(list)
    recovered_count = 0

    for record in records:
        if record.get("dataset_sha256") != digest:
            raise ValueError("Dataset fingerprint mismatch")

        cid = record.get("candidate")
        dimension = record.get("dimension")

        if cid not in ids or dimension not in DIMENSIONS:
            raise ValueError("Unknown candidate or dimension")

        if record.get("recovered_from_archived_response"):
            if not record.get("recovery_corrections"):
                raise ValueError("Recovery provenance missing")
            recovered_count += 1

        if record.get("valid") is True:
            verdict = record.get("verdict")
            if not isinstance(verdict, dict):
                raise ValueError("Missing valid verdict")
            score = verdict.get("score")
            if type(score) is not int or not 1 <= score <= 10:
                raise ValueError("Invalid dimension score")
            if not isinstance(verdict.get("reason"), str):
                raise ValueError("Missing dimension reason")

        grouped[(cid, dimension)].append(record)

    profiles = []

    for cid in sorted(ids):
        dimensions = {}
        weighted = 0.0
        complete = True

        for dimension, weight in DIMENSIONS.items():
            rows = grouped[(cid, dimension)]
            valid_rows = [
                row for row in rows if row.get("valid") is True
            ]
            scores = [
                row["verdict"]["score"]
                for row in valid_rows
            ]

            if not scores:
                complete = False

            average = mean(scores) if scores else None
            if average is not None:
                weighted += average * weight / 100

            dimensions[dimension] = {
                "weight": weight,
                "evaluations": len(rows),
                "valid": len(scores),
                "mean_score": round(average, 3)
                if average is not None else None,
                "stddev": round(pstdev(scores), 3)
                if len(scores) > 1 else None,
                "judgments": [
                    {
                        "score": row["verdict"]["score"],
                        "reason": row["verdict"]["reason"],
                        "evidence": row["verdict"].get("evidence", []),
                        "recovered": bool(
                            row.get("recovered_from_archived_response")
                        ),
                        "recovery_corrections": row.get(
                            "recovery_corrections", []
                        ),
                    }
                    for row in valid_rows
                ],
            }

        profiles.append({
            "candidate": cid,
            "complete": complete,
            "weighted_score": round(weighted, 3)
            if complete else None,
            "dimensions": dimensions,
        })

    return {
        "dataset_sha256": digest,
        "results_file": str(results_file),
        "evaluations": len(records),
        "valid": sum(r.get("valid") is True for r in records),
        "invalid": sum(r.get("valid") is not True for r in records),
        "recovered": recovered_count,
        "weights": DIMENSIONS,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    report = build_report(args.dataset, args.results)
    content = json.dumps(
        report, indent=2, ensure_ascii=False
    ) + "\n"

    if args.out:
        destination = Path(args.out)
        if destination.exists():
            parser.error(f"Output already exists: {destination}")
        destination.write_text(
            content, encoding="utf-8"
        )
        print("Saved:", destination)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
