"""End-to-end ST text completion self-play and resume."""

import json
from pathlib import Path

import pytest

from rpbench.core import load
from rpbench.simulation import run_simulation
from rpbench.resume import resume_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/character-chemistry-de/scenario.json"
PLAYER = ROOT / "scenarios/character-chemistry-de/player.json"


def write_config(tmp_path):
    preset = {
        "id": "st-test-v1",
        "style": "st_instruct_v1",
        "context_preset": {
            "story_string": "{{system}}\n{{trim}}",
            "story_string_position": 0,
            "chat_start": "",
        },
        "instruct_preset": {
            "input_sequence": "[USER]",
            "output_sequence": "[ASSISTANT]",
            "input_suffix": "[END]",
            "output_suffix": "[END]",
            "wrap": False,
            "macro": False,
            "names_behavior": "force",
        },
    }

    preset_path = tmp_path / "st-preset.json"
    preset_path.write_text(
        json.dumps(preset), encoding="utf-8"
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "mode": "completion",
            "model": "mock",
        },
        "sampling": {"max_tokens": 128},
        "completion_preset": "st-preset.json",
    }), encoding="utf-8")

    return config_path, preset_path


def requests(root, role):
    return [
        json.loads(line)["payload"]
        for line in (root / "events.jsonl").read_text().splitlines()
        if (
            (event := json.loads(line)).get("type") == "request"
            and event.get("role") == role
        )
    ]


def test_selfplay_and_resume_with_archived_st_preset(tmp_path):
    config, preset = write_config(tmp_path)

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    manifest = load(root / "manifest.json")
    assert manifest["completion_preset"]["id"] == "st-test-v1"
    assert manifest["prompt_format"]["style"] == "st_instruct_v1"
    assert manifest["player_prompt_format"]["style"] == "st_instruct_v1"

    npc_before = requests(root, "npc")
    player_before = requests(root, "player")

    assert len(npc_before) == 1
    assert len(player_before) == 1

    assert npc_before[0]["messages"][0]["role"] == "system"
    assert npc_before[0]["messages"][0]["content"] == (
        manifest["rendered_npc_system_prompt"]
    )

    # Mock records structured messages; the completion renderer
    # is verified separately using Backend.prepare().
    npc_name = manifest["prompt_format"]["character_name"]
    player_name = manifest["player_prompt_format"]["character_name"]
    assert npc_name != player_name

    # Resume must not depend on the source preset.
    preset.unlink()
    resume_simulation(root, turns=2)

    after = load(root / "manifest.json")
    assert after["status"] == "complete"
    assert after["completed_turns"] == 2

    assert len(requests(root, "npc")) == 2
    assert len(requests(root, "player")) == 2
    assert after["prompt_format"] == manifest["prompt_format"]
    assert after["player_prompt_format"] == (
        manifest["player_prompt_format"]
    )


def test_wrong_backend_mode_rejected(tmp_path):
    config, _ = write_config(tmp_path)
    data = json.loads(config.read_text())
    data["backend"]["mode"] = "chat"
    config.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="completion mode"):
        run_simulation(
            SCENARIO, PLAYER, config,
            tmp_path / "runs", turns=1,
        )

    assert not (tmp_path / "runs").exists()


def test_real_completion_payload_and_resume(tmp_path, monkeypatch):
    """Exercise actual Backend.prepare, without network inference."""
    config_path, preset_path = write_config(tmp_path)
    config = json.loads(config_path.read_text())

    config["backend"] = {
        "name": "llamacpp",
        "mode": "completion",
        "model": "test",
        "base_url": "http://127.0.0.1:8080",
    }
    config_path.write_text(json.dumps(config))

    captured = []

    def fake_generate(self, url, payload):
        captured.append((url, payload))
        return "A fictional reply.", {
            "choices": [{
                "text": "A fictional reply.",
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(
        "rpbench.simulation.Backend.generate",
        fake_generate,
    )

    root = run_simulation(
        SCENARIO, PLAYER, config_path,
        tmp_path / "runs", turns=1,
    )

    assert len(captured) == 2

    player_prompt = captured[0][1]["prompt"]
    npc_prompt = captured[1][1]["prompt"]

    assert "[USER]" in npc_prompt
    assert npc_prompt.endswith("[ASSISTANT]")
    assert "CHARACTER DEFINITIONS:" in npc_prompt
    assert "ST TEST PREFIX" not in npc_prompt

    assert player_prompt.endswith("[ASSISTANT]")
    assert player_prompt != npc_prompt

    preset_path.unlink()
    resume_simulation(root, turns=2)

    assert len(captured) == 4

    resumed_npc = captured[3][1]["prompt"]
    assert resumed_npc.startswith(
        npc_prompt.removesuffix("[ASSISTANT]")
    )

    manifest = load(root / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["completed_turns"] == 2
