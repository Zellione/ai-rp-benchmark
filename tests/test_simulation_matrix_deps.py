import json

import pytest

from rpbench.simulation_matrix_deps import dependency_digests


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def test_character_card_changes_dependency_hash(tmp_path):
    scenario = tmp_path / "scenario.json"
    config = tmp_path / "config.json"
    card = tmp_path / "character.json"

    put(scenario, {
        "system": "NPC",
        "character_cards": ["character.json"],
    })
    put(config, {"backend": {"model": "test"}})
    put(card, {"name": "Mara"})

    before = dependency_digests(scenario, config)
    put(card, {"name": "Mara", "personality": "Suspicious"})
    after = dependency_digests(scenario, config)

    assert before != after


@pytest.mark.parametrize(
    "setting,field",
    [
        ("prompt_profile", "prompt_profile"),
        ("completion_preset", "completion_preset"),
    ],
)
def test_config_reference_changes_hash(tmp_path, setting, field):
    scenario = tmp_path / "scenario.json"
    config = tmp_path / "config.json"
    dependency = tmp_path / "reference.json"

    put(scenario, {"system": "NPC"})
    put(config, {setting: "reference.json"})
    put(dependency, {"version": 1})

    before = dependency_digests(scenario, config)
    put(dependency, {"version": 2})
    after = dependency_digests(scenario, config)

    assert field in before
    assert before != after


def test_st_preset_changes_dependency_hash(tmp_path):
    scenario = tmp_path / "scenario.json"
    config = tmp_path / "config.json"
    preset = tmp_path / "preset.json"

    put(scenario, {"system": "NPC"})
    put(config, {
        "runtime": {
            "st_chat_npc": {"preset": "preset.json"}
        }
    })
    put(preset, {"prompts": [1]})

    before = dependency_digests(scenario, config)
    put(preset, {"prompts": [1, 2]})
    after = dependency_digests(scenario, config)

    assert before["st_chat_npc:preset"] != after["st_chat_npc:preset"]


def test_missing_dependency_fails_preflight(tmp_path):
    scenario = tmp_path / "scenario.json"
    config = tmp_path / "config.json"

    put(scenario, {"system": "NPC"})
    put(config, {"prompt_profile": "missing.json"})

    with pytest.raises(ValueError, match="Missing file"):
        dependency_digests(scenario, config)
