"""Narrative player mode and external backend survive resume."""

import json
from pathlib import Path

from rpbench.core import Backend, load
from rpbench.simulation import run_simulation
from rpbench.resume import resume_simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def test_narrative_player_survives_resume(tmp_path, monkeypatch):
    prompts = []

    def fake_generate(self, url, payload):
        system = payload["messages"][0]["content"]

        if self.model == "external-player-test":
            prompts.append(system)
            response = "*Alex studies the room.*"
        else:
            response = "Mara considers the visitor's question."

        return response, {
            "choices": [{
                "message": {"content": response},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "mode": "chat",
        },
        "player_backend": {
            "name": "nanogpt",
            "base_url": "https://nano-gpt.com/api",
            "model": "external-player-test",
            "mode": "chat",
            "api_key_env": "NANOGPT_API_KEY",
        },
        "runtime": {
            "controlled_player": True,
            "player_control_mode": "narrative",
            "player_knowledge_rules": [],
        },
    }), encoding="utf-8")

    run = run_simulation(
        SCENARIO, PLAYER, config_path,
        tmp_path / "runs", turns=1,
    )

    assert len(prompts) == 1
    assert "CONTROLLED PLAYER NARRATIVE DIRECTION:" in prompts[0]
    assert "one concrete next action" not in prompts[0]

    manifest = load(run / "manifest.json")

    assert manifest["runtime"]["player_control_mode"] == "narrative"
    assert (
        manifest["player_backend"]["api_key_env"]
        == "NANOGPT_API_KEY"
    )

    resume_simulation(run, turns=2)

    assert len(prompts) == 2
    assert "CONTROLLED PLAYER NARRATIVE DIRECTION:" in prompts[1]
    assert "one concrete next action" not in prompts[1]

    final = load(run / "manifest.json")
    assert final["status"] == "complete"
    assert final["completed_turns"] == 2
    assert final["player_backend"]["model"] == "external-player-test"
    assert (
        final["player_backend"]["api_key_env"]
        == "NANOGPT_API_KEY"
    )
