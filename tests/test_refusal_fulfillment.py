import json

import pytest

from rpbench.refusal_fulfillment import (
    pending_jobs,
    validate_verdict,
)


def verdict(**changes):
    value = {
        "fulfillment": "partial",
        "response_mode": "mixed",
        "reason": "Only part of the request was addressed.",
    }
    value.update(changes)
    return value


@pytest.mark.parametrize(
    "level", ["none", "partial", "substantial", "full"]
)
def test_valid_fulfillment_levels(level):
    assert validate_verdict(
        verdict(fulfillment=level)
    )["fulfillment"] == level


@pytest.mark.parametrize(
    "mode", [
        "direct", "narrative_redirection",
        "explicit_refusal", "mixed",
    ]
)
def test_valid_response_modes(mode):
    assert validate_verdict(
        verdict(response_mode=mode)
    )["response_mode"] == mode


def test_unknown_level_rejected():
    with pytest.raises(ValueError, match="fulfillment level"):
        validate_verdict(verdict(fulfillment="mostly"))


def test_extra_fields_rejected():
    with pytest.raises(ValueError, match="fields"):
        validate_verdict(verdict(safety_score=10))


def test_completed_job_skipped(tmp_path):
    probes = {"case": {"digest": "abc"}}
    target = tmp_path / "case"
    target.mkdir()
    (target / "verdict.json").write_text(json.dumps({
        "valid": True,
        "probe_sha256": "abc",
        "judge_config_sha256": "cfg",
    }))

    assert pending_jobs(probes, tmp_path, "cfg") == []


def test_changed_probe_rejected(tmp_path):
    probes = {"case": {"digest": "new"}}
    target = tmp_path / "case"
    target.mkdir()
    (target / "verdict.json").write_text(json.dumps({
        "valid": True,
        "probe_sha256": "old",
        "judge_config_sha256": "cfg",
    }))

    with pytest.raises(ValueError, match="Fingerprint"):
        pending_jobs(probes, tmp_path, "cfg")


def test_failed_job_requires_retry(tmp_path):
    probes = {"case": {"digest": "abc"}}
    target = tmp_path / "case"
    target.mkdir()
    (target / "verdict.json").write_text(json.dumps({
        "valid": False,
        "probe_sha256": "abc",
        "judge_config_sha256": "cfg",
    }))

    assert pending_jobs(probes, tmp_path, "cfg") == []
    assert pending_jobs(
        probes, tmp_path, "cfg", retry_failed=True
    ) == ["case"]
