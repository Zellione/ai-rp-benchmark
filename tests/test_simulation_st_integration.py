"""End-to-end opt-in SillyTavern NPC rendering tests."""

import json
from pathlib import Path

import pytest

from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def write_config(tmp_path, *, enabled=True):
    preset_path = tmp_path / "preset.json"
    preset = {
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "ST TEST PREFIX {{char}}",
            },
            {
                "identifier": "rpbenchContext",
                "role": "system",
                "marker": True,
            },
            {
                "identifier": "chatHistory",
                "role": "system",
                "marker": True,
            },
        ],
        "prompt_order": [{
            "character_id": 100000,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "rpbenchContext", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    }

    preset_path.write_text(json.dumps(preset))

    config = {
        "backend": {"name": "mock", "mode": "chat"},
        "sampling": {},
    }

    if enabled:
        config["runtime"] = {
            "st_chat_npc": {
                "preset": "preset.json",
                "character_name": "Elara",
                "user_name": "Rowan",
                "context_marker": "rpbenchContext",
            },
        }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    return config_path, preset_path


def requests(run_dir, role):
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    return [
        event["payload"]["messages"]
        for event in events
        if event["type"] == "request"
        and event.get("role") == role
    ]


def test_opt_in_simulation_and_resume(tmp_path):
    config_path, preset_path = write_config(tmp_path)

    run_dir = run_simulation(
        SCENARIO,
        PLAYER,
        config_path,
        tmp_path / "runs",
        turns=1,
    )

    manifest = json.loads(
        (run_dir / "manifest.json").read_text()
    )

    assert manifest["npc_renderer"]["mode"] == (
        "sillytavern_chat"
    )

    original_npc = requests(run_dir, "npc")
    assert len(original_npc) == 1

    assert original_npc[0][0] == {
        "role": "system",
        "content": "ST TEST PREFIX Elara",
    }

    assert any(
        "CHARACTER DEFINITIONS:" in m["content"]
        for m in original_npc[0]
    )

    assert "ST TEST PREFIX" not in json.dumps(
        requests(run_dir, "player")
    )

    # Resume must use the archived preset, not the source file.
    preset_path.unlink()

    resume_simulation(run_dir, turns=2)

    resumed_npc = requests(run_dir, "npc")
    assert len(resumed_npc) == 2

    assert resumed_npc[1][0] == original_npc[0][0]

    final_manifest = json.loads(
        (run_dir / "manifest.json").read_text()
    )

    assert final_manifest["status"] == "complete"
    assert final_manifest["completed_turns"] == 2


def test_legacy_simulation_has_no_renderer(tmp_path):
    config_path, _ = write_config(tmp_path, enabled=False)

    run_dir = run_simulation(
        SCENARIO,
        PLAYER,
        config_path,
        tmp_path / "runs",
        turns=1,
    )

    manifest = json.loads(
        (run_dir / "manifest.json").read_text()
    )

    assert "npc_renderer" not in manifest

    npc = requests(run_dir, "npc")
    assert npc[0][0]["content"] == (
        manifest["rendered_npc_system_prompt"]
    )


def test_invalid_preset_rejected_before_run(tmp_path):
    config_path, preset_path = write_config(tmp_path)
    preset = json.loads(preset_path.read_text())

    preset["prompts"][1]["marker"] = False
    preset_path.write_text(json.dumps(preset))

    runs = tmp_path / "runs"

    with pytest.raises(ValueError):
        run_simulation(
            SCENARIO,
            PLAYER,
            config_path,
            runs,
            turns=1,
        )

    assert not runs.exists()
