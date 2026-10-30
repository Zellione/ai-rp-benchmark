import json

import pytest

from rpbench.simulation_matrix import execute, plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path):
    for language in ("de", "en"):
        write(
            tmp_path / f"{language}-scenario.json",
            {"id": f"scenario-{language}", "system": "NPC"},
        )
        write(
            tmp_path / f"{language}-player.json",
            {"name": "Player"},
        )

    for preset in ("baseline", "preset-a"):
        write(
            tmp_path / f"{preset}.json",
            {"backend": {"model": "test-model"}},
        )

    matrix = {
        "schema_version": 1,
        "model": {
            "id": "test-model",
            "backend_model": "test-model",
        },
        "turns": 5,
        "scenarios": [
            {
                "id": "scene",
                "language": language,
                "scenario": f"{language}-scenario.json",
                "player": f"{language}-player.json",
            }
            for language in ("de", "en")
        ],
        "presets": [
            {"id": preset, "config": f"{preset}.json"}
            for preset in ("baseline", "preset-a")
        ],
    }
    path = tmp_path / "matrix.json"
    write(path, matrix)
    return path


def test_bilingual_preset_cartesian_product(tmp_path):
    jobs = plan(fixture(tmp_path))
    assert len(jobs) == 4
    assert {
        (j["identity"]["language"], j["identity"]["preset"])
        for j in jobs
    } == {
        ("de", "baseline"), ("de", "preset-a"),
        ("en", "baseline"), ("en", "preset-a"),
    }


def test_single_model_validation(tmp_path):
    path = fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["model"]["backend_model"] = "another-model"
    write(path, matrix)

    with pytest.raises(ValueError, match="Wrong model"):
        plan(path)


def test_dry_run_does_not_write(tmp_path):
    jobs = plan(fixture(tmp_path))
    output = tmp_path / "runs"

    execute(jobs, output, dry_run=True)
    assert not output.exists()


def test_input_change_changes_identity(tmp_path):
    path = fixture(tmp_path)
    before = plan(path)[0]["key"]

    scenario = tmp_path / "de-scenario.json"
    write(scenario, {"id": "modified", "system": "NPC"})

    after = plan(path)[0]["key"]
    assert before != after
