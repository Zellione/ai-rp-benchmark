import json

import pytest

from rpbench.simulation_matrix import plan


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def matrix_fixture(tmp_path):
    for language in ("de", "en"):
        write(
            tmp_path / f"scenario-{language}.json",
            {"id": "test", "language": language, "system": "NPC"},
        )
        write(
            tmp_path / f"player-{language}.json",
            {"name": "Alex"},
        )
        write(
            tmp_path / f"config-{language}.json",
            {"backend": {"model": "model-1"}},
        )

    matrix = {
        "schema_version": 1,
        "model": {
            "id": "test-model",
            "backend_model": "model-1",
        },
        "turns": 3,
        "scenarios": [
            {
                "id": "test",
                "language": language,
                "scenario": f"scenario-{language}.json",
                "player": f"player-{language}.json",
            }
            for language in ("de", "en")
        ],
        "presets": [{
            "id": "megumin",
            "configs_by_language": {
                "de": "config-de.json",
                "en": "config-en.json",
            },
        }],
    }

    path = tmp_path / "matrix.json"
    write(path, matrix)
    return path


def test_language_specific_configs(tmp_path):
    jobs = plan(matrix_fixture(tmp_path))
    assert len(jobs) == 2

    by_language = {
        job["identity"]["language"]: job
        for job in jobs
    }

    assert by_language["de"]["config_path"].name == "config-de.json"
    assert by_language["en"]["config_path"].name == "config-en.json"
    assert by_language["de"]["key"] != by_language["en"]["key"]


def test_missing_language_config_rejected(tmp_path):
    path = matrix_fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["presets"][0]["configs_by_language"].pop("en")
    write(path, matrix)

    with pytest.raises(ValueError, match="de/en"):
        plan(path)


def test_ambiguous_config_rejected(tmp_path):
    path = matrix_fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["presets"][0]["config"] = "config-de.json"
    write(path, matrix)

    with pytest.raises(ValueError, match="exactly"):
        plan(path)
