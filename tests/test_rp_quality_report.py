import hashlib
import json

import pytest

from rpbench.rp_quality_judge import DIMENSIONS
from rpbench.rp_quality_report import build_report


def fixture_files(tmp_path, *, missing=None, recovered=False):
    dataset_file = tmp_path / "dataset.json"
    results_file = tmp_path / "results.jsonl"

    dataset = {
        "schema_version": 1,
        "cases": [{"id": "A"}, {"id": "B"}],
    }
    dataset_file.write_text(json.dumps(dataset), encoding="utf-8")

    digest = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    scores = {
        "A": {
            "german_prose": 7,
            "character_fidelity": 8,
            "dialogue_responsiveness": 8,
            "subtext": 8,
            "narrative_initiative": 8,
            "continuity": 7,
        },
        "B": {key: 6 for key in DIMENSIONS},
    }

    records = []
    for candidate, dimensions in scores.items():
        for dimension, score in dimensions.items():
            if (candidate, dimension) == missing:
                continue
            row = {
                "dataset_sha256": digest,
                "candidate": candidate,
                "dimension": dimension,
                "valid": True,
                "verdict": {
                    "score": score,
                    "reason": "Grounded explanation",
                    "evidence": [
                        {"turn": 1, "quote": "NPC excerpt"}
                    ],
                },
            }
            if recovered and candidate == "A" and dimension == "continuity":
                row["recovered_from_archived_response"] = True
                row["recovery_corrections"] = [{
                    "original": {"turn": 2},
                    "replacement": {"turn": 1},
                }]
            records.append(row)

    results_file.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return dataset_file, results_file


def test_weighted_scores(tmp_path):
    dataset, results = fixture_files(tmp_path)
    report = build_report(dataset, results)

    profiles = {
        profile["candidate"]: profile
        for profile in report["profiles"]
    }

    assert report["valid"] == 12
    assert report["invalid"] == 0
    assert profiles["A"]["weighted_score"] == 7.65
    assert profiles["B"]["weighted_score"] == 6.0


def test_missing_dimension_prevents_total(tmp_path):
    dataset, results = fixture_files(
        tmp_path, missing=("A", "continuity")
    )
    report = build_report(dataset, results)
    profile = next(
        p for p in report["profiles"] if p["candidate"] == "A"
    )

    assert profile["complete"] is False
    assert profile["weighted_score"] is None


def test_fingerprint_mismatch_rejected(tmp_path):
    dataset, results = fixture_files(tmp_path)
    payload = json.loads(dataset.read_text())
    payload["version"] = 2
    dataset.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fingerprint"):
        build_report(dataset, results)


def test_recovery_provenance(tmp_path):
    dataset, results = fixture_files(tmp_path, recovered=True)
    report = build_report(dataset, results)

    assert report["recovered"] == 1
    profile = next(
        p for p in report["profiles"] if p["candidate"] == "A"
    )
    assert profile["dimensions"]["continuity"][
        "judgments"
    ][0]["recovered"] is True


def test_invalid_recovery_without_provenance(tmp_path):
    dataset, results = fixture_files(tmp_path)
    records = [
        json.loads(line)
        for line in results.read_text().splitlines()
    ]
    records[0]["recovered_from_archived_response"] = True
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )

    with pytest.raises(ValueError, match="provenance"):
        build_report(dataset, results)
