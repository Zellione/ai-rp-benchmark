import json

import pytest

from rpbench.refusal_baseline_report import (
    build_report,
    summarize_deployment,
)


IDENTITY = {
    "id": "test/Model",
    "base_model": "Model",
    "publisher": "test",
    "variant": "standard",
    "quantization": "Q6_K",
    "source": {"type": "local", "path": "mock://model"},
}


def write_attempt(root, scenario, status, number=1):
    folder = root / "batches/test-deployment" / f"attempt-{number}-{scenario}"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({
        "scenario_id": scenario,
        "category": "benign_trigger",
        "status": status,
        "model_identity": IDENTITY,
        "backend": {"name": "mock", "model": "mock", "mode": "chat"},
    }))


def test_complete_generation_without_judgments(tmp_path):
    write_attempt(tmp_path, "case-a", "complete")

    report = summarize_deployment(
        tmp_path, "test-deployment", {"case-a"}
    )

    assert report["generated"] == 1
    assert report["refusal_evaluated"] == 0
    assert report["fulfillment_evaluated"] == 0
    assert report["model_refusals"] == 0
    assert report["cases"][0]["refusal"] is None


def test_truncation_is_not_refusal(tmp_path):
    write_attempt(tmp_path, "case-a", "truncated")

    report = summarize_deployment(
        tmp_path, "test-deployment", {"case-a"}
    )

    assert report["generated"] == 0
    assert report["generation_statuses"] == {"truncated": 1}
    assert report["cases"][0]["fulfillment"] is None
    assert report["cases"][0]["refusal"] is None


def test_retry_preserved(tmp_path):
    write_attempt(tmp_path, "case-a", "failed", 1)
    write_attempt(tmp_path, "case-a", "complete", 2)

    report = summarize_deployment(
        tmp_path, "test-deployment", {"case-a"}
    )

    assert report["generated"] == 1
    assert report["generation_attempts"] == 2


def test_duplicate_complete_rejected(tmp_path):
    write_attempt(tmp_path, "case-a", "complete", 1)
    write_attempt(tmp_path, "case-a", "complete", 2)

    with pytest.raises(ValueError, match="Multiple completed"):
        summarize_deployment(
            tmp_path, "test-deployment", {"case-a"}
        )


def test_manifest_coverage_rejected(tmp_path):
    write_attempt(tmp_path, "case-a", "complete")

    with pytest.raises(ValueError, match="coverage mismatch"):
        summarize_deployment(
            tmp_path, "test-deployment", {"case-a", "case-b"}
        )


def test_manifest_report(tmp_path):
    write_attempt(tmp_path, "case-a", "complete")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "cases": [{"id": "case-a"}]
    }))

    report = build_report(tmp_path, manifest)
    assert report["deployment_count"] == 1
    assert report["scenario_ids"] == ["case-a"]
