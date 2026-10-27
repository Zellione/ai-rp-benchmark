"""Fingerprint integrity and distinct semantic accuracy metrics."""

import hashlib
import json
from pathlib import Path

import pytest

from rpbench.semantic_benchmark_report import report


def write_json(filename, value):
    filename.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def make_files(tmp_path, *, fingerprint=True):
    suite = {
        "schema_version": 1,
        "suite_id": "test-suite",
        "cases": [{
            "id": "partial-case",
            "goal_id": "g1",
            "criterion_id": "c1",
            "description": "A restricted goal",
            "expected_status": "not_satisfied",
            "observations": [
                {
                    "turn": 1,
                    "role": "player",
                    "text": "Hello",
                },
                {
                    "turn": 1,
                    "role": "npc",
                    "text": "No authorization.",
                },
            ],
            "prior_partials": [],
        }],
    }
    suite_file = tmp_path / "suite.json"
    results_file = tmp_path / "results.jsonl"
    write_json(suite_file, suite)

    entry = {
        "suite_id": "test-suite",
        "case": "partial-case",
        "valid": True,
        "status": "partial",
        "source_turns": [1],
        "usage": {"completion_tokens": 100},
        "elapsed_seconds": 3.5,
        "reasoning_only_truncation": False,
    }

    if fingerprint:
        entry["suite_sha256"] = hashlib.sha256(
            suite_file.read_bytes()
        ).hexdigest()

    results_file.write_text(
        json.dumps(entry) + "\n",
        encoding="utf-8",
    )
    return suite_file, results_file


def test_partial_error_does_not_falsely_complete(tmp_path):
    suite, results = make_files(tmp_path)
    result = report(suite, results)

    assert result["evaluations"] == 1
    assert result["valid"] == 1
    assert result["status_correct"] == 0
    assert result["completion_correct"] == 1
    assert result["false_completions"] == 0
    assert result["missed_completions"] == 0
    assert result["completion_tokens"] == 100


def test_modified_suite_rejected(tmp_path):
    suite_file, results = make_files(tmp_path)

    suite = json.loads(suite_file.read_text())
    suite["cases"][0]["expected_status"] = "partial"
    write_json(suite_file, suite)

    with pytest.raises(ValueError, match="fingerprint"):
        report(suite_file, results)


def test_legacy_requires_explicit_permission(tmp_path):
    suite, results = make_files(
        tmp_path, fingerprint=False
    )

    with pytest.raises(ValueError, match="fingerprint"):
        report(suite, results)

    summary = report(suite, results, allow_legacy=True)
    assert summary["legacy_results"] is True


def test_unrecognized_case_rejected(tmp_path):
    suite, results = make_files(tmp_path)
    entry = json.loads(results.read_text())
    entry["case"] = "missing-case"
    results.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="Unknown"):
        report(suite, results)
