import json

import pytest

from rpbench.refusal_compare import summarize


IDENTITY = {
    "id": "example/Model",
    "quantization": "Q4_K_M",
}


def write_probe(root, name, status, attempt):
    directory = root / f"{attempt}_{name}"
    directory.mkdir(parents=True)

    (directory / "result.json").write_text(
        json.dumps({
            "scenario_id": name,
            "category": "dark_fiction",
            "status": status,
            "model_identity": IDENTITY,
        }),
        encoding="utf-8",
    )


def write_judgment(root, name, **changes):
    directory = root / name
    directory.mkdir(parents=True)

    verdict = {
        "model_refusal": False,
        "refusal_style": "none",
        "roleplay_continuity": True,
        "unsafe_actionable": False,
        "unsafe_sexual_content": False,
    }
    verdict.update(changes)

    (directory / "verdict.json").write_text(
        json.dumps({
            "scenario_id": name,
            "valid": True,
            "model_identity": IDENTITY,
            "verdict": verdict,
        }),
        encoding="utf-8",
    )


def test_successful_deployment(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "complete", 1)
    write_judgment(judges, "first")

    report = summarize(probes, judges)

    assert report["scenario_count"] == 1
    assert report["generated_scenarios"] == 1
    assert report["judged_scenarios"] == 1
    assert report["model_refusals"] == 0
    assert report["rp_continuations"] == 1


def test_repeated_empty_response_preserved(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "horror", "empty_response", 1)
    write_probe(probes, "horror", "empty_response", 2)

    report = summarize(probes, judges)

    assert report["scenario_count"] == 1
    assert report["generation_attempts"] == 2
    assert report["generated_scenarios"] == 0
    assert report["judged_scenarios"] == 0
    assert report["model_refusals"] == 0
    assert report["generation_statuses"] == {
        "empty_response": 2
    }
    assert report["unjudged_scenarios"] == ["horror"]


def test_failed_attempt_followed_by_success(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "empty_response", 1)
    write_probe(probes, "first", "complete", 2)
    write_judgment(judges, "first")

    report = summarize(probes, judges)

    assert report["generation_attempts"] == 2
    assert report["generated_scenarios"] == 1
    assert report["judged_scenarios"] == 1


def test_duplicate_completed_attempts_rejected(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "complete", 1)
    write_probe(probes, "first", "complete", 2)

    with pytest.raises(ValueError, match="Multiple completed"):
        summarize(probes, judges)


def test_mixed_model_identity_rejected(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "complete", 1)
    write_probe(probes, "second", "complete", 2)

    file = probes / "2_second" / "result.json"
    record = json.loads(file.read_text())
    record["model_identity"]["id"] = "other/Model"
    file.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="Mixed model"):
        summarize(probes, judges)


def test_judgment_without_probe_rejected(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "complete", 1)
    write_judgment(judges, "other")

    with pytest.raises(ValueError, match="without probe"):
        summarize(probes, judges)


def test_judgment_for_failed_probe_rejected(tmp_path):
    probes = tmp_path / "probes"
    judges = tmp_path / "judgments"

    write_probe(probes, "first", "empty_response", 1)
    write_judgment(judges, "first")

    with pytest.raises(ValueError, match="incomplete probe"):
        summarize(probes, judges)
