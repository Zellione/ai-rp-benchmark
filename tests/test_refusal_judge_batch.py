import json

import pytest

from rpbench.refusal_judge_batch import (
    load_probes,
    pending_probes,
    read_attempts,
)


def make_probes(tmp_path):
    directory = tmp_path / "probes"

    for scenario in ("first", "second"):
        folder = directory / f"run-{scenario}"
        folder.mkdir(parents=True)

        (folder / "result.json").write_text(
            json.dumps({
                "scenario_id": scenario,
                "status": "complete",
                "model_identity": {"id": "example/model"},
            }),
            encoding="utf-8",
        )

    return directory


def test_load_completed_probes(tmp_path):
    probes = load_probes(make_probes(tmp_path))

    assert set(probes) == {"first", "second"}
    assert all(len(item["digest"]) == 64 for item in probes.values())


def test_valid_judgments_are_skipped(tmp_path):
    probes = load_probes(make_probes(tmp_path))
    attempts = {
        "first": [{"valid": True}],
    }

    assert pending_probes(probes, attempts) == ["second"]


def test_failed_judgment_requires_retry(tmp_path):
    probes = load_probes(make_probes(tmp_path))
    attempts = {
        "first": [{"valid": False}],
    }

    assert pending_probes(probes, attempts) == ["second"]
    assert pending_probes(
        probes, attempts, retry_failed=True
    ) == ["first", "second"]


def test_duplicate_probe_rejected(tmp_path):
    directory = make_probes(tmp_path)
    duplicate = directory / "duplicate"
    duplicate.mkdir()

    original = directory / "run-first" / "result.json"
    (duplicate / "result.json").write_bytes(
        original.read_bytes()
    )

    with pytest.raises(ValueError, match="Multiple probe"):
        load_probes(directory)


def test_changed_judge_config_rejected(tmp_path):
    probes = load_probes(make_probes(tmp_path))
    output = tmp_path / "judgments" / "first"
    output.mkdir(parents=True)

    (output / "attempt.json").write_text(
        json.dumps({
            "scenario_id": "first",
            "probe_sha256": probes["first"]["digest"],
            "judge_config_sha256": "original",
            "valid": True,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration changed"):
        read_attempts(
            tmp_path / "judgments", probes, "modified"
        )


def test_changed_probe_rejected(tmp_path):
    probes = load_probes(make_probes(tmp_path))
    output = tmp_path / "judgments" / "first"
    output.mkdir(parents=True)

    (output / "attempt.json").write_text(
        json.dumps({
            "scenario_id": "first",
            "probe_sha256": "wrong",
            "judge_config_sha256": "same",
            "valid": True,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Probe content changed"):
        read_attempts(
            tmp_path / "judgments", probes, "same"
        )
