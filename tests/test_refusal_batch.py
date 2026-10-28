import json

import pytest

from rpbench.refusal_batch import (
    archived_attempts,
    deployment_name,
    load_plan,
    pending_plan,
)
from tools.build_refusal_pilot import generate


def setup_batch(tmp_path):
    scenario_dir = tmp_path / "scenarios"
    generate(scenario_dir)

    config = {
        "backend": {
            "name": "mock",
            "base_url": "",
            "model": "mock",
            "mode": "chat",
        },
        "sampling": {"max_tokens": 256},
        "model_identity": {
            "id": "example/Test-Model",
            "base_model": "Test-Model",
            "publisher": "example",
            "variant": "standard",
            "quantization": "Q6_K_M",
            "source": {
                "type": "local",
                "path": "mock://model",
            },
        },
    }

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")

    manifest_file = scenario_dir / "manifest.json"
    return manifest_file, config_file, config


def test_plan_contains_sixteen_probes(tmp_path):
    manifest, config, _ = setup_batch(tmp_path)
    plan, identity, digest = load_plan(manifest, config)

    assert len(plan) == 16
    assert identity["id"] == "example/Test-Model"
    assert len(digest) == 64


def test_deployment_name_includes_quantization():
    identity = {
        "id": "unsloth/Qwen3.8-27B",
        "variant": "UD",
        "quantization": "Q6_K_M",
    }
    assert deployment_name(identity) == (
        "unsloth_Qwen3.8-27B__UD__Q6_K_M"
    )


def test_completed_job_is_skipped(tmp_path):
    manifest, config, _ = setup_batch(tmp_path)
    plan, _, _ = load_plan(manifest, config)

    archived = {
        plan[0]["id"]: [{
            "status": "complete",
            "scenario_sha256": plan[0]["scenario_sha256"],
        }],
    }

    pending = pending_plan(plan, archived)
    assert len(pending) == 15
    assert plan[0] not in pending


def test_failed_job_requires_explicit_retry(tmp_path):
    manifest, config, _ = setup_batch(tmp_path)
    plan, _, _ = load_plan(manifest, config)

    archived = {
        plan[0]["id"]: [{
            "status": "failed",
            "scenario_sha256": plan[0]["scenario_sha256"],
        }],
    }

    assert len(pending_plan(plan, archived)) == 15
    assert len(
        pending_plan(plan, archived, retry_failed=True)
    ) == 16


def test_changed_scenario_rejected(tmp_path):
    manifest, config, _ = setup_batch(tmp_path)
    plan, _, _ = load_plan(manifest, config)

    archived = {
        plan[0]["id"]: [{
            "status": "complete",
            "scenario_sha256": "incorrect",
        }],
    }

    with pytest.raises(ValueError, match="Scenario changed"):
        pending_plan(plan, archived)


def test_changed_config_rejected(tmp_path):
    manifest, config, cfg = setup_batch(tmp_path)
    plan, identity, digest = load_plan(manifest, config)

    directory = tmp_path / "runs"
    attempt = directory / "attempt-1"
    attempt.mkdir(parents=True)

    (attempt / "result.json").write_text(
        json.dumps({
            "scenario_id": plan[0]["id"],
            "model_identity": identity,
            "config_sha256": digest,
            "scenario_sha256": plan[0]["scenario_sha256"],
            "status": "complete",
        }),
        encoding="utf-8",
    )

    assert len(archived_attempts(directory, identity, digest)) == 1

    with pytest.raises(ValueError, match="Config mismatch"):
        archived_attempts(directory, identity, "different")


def test_model_identity_mismatch_rejected(tmp_path):
    manifest, config, _ = setup_batch(tmp_path)
    plan, identity, digest = load_plan(manifest, config)

    directory = tmp_path / "runs"
    attempt = directory / "attempt-1"
    attempt.mkdir(parents=True)

    (attempt / "result.json").write_text(
        json.dumps({
            "scenario_id": plan[0]["id"],
            "model_identity": {
                **identity,
                "quantization": "NVFP4",
            },
            "config_sha256": digest,
            "scenario_sha256": plan[0]["scenario_sha256"],
            "status": "complete",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Model identity mismatch"):
        archived_attempts(directory, identity, digest)
