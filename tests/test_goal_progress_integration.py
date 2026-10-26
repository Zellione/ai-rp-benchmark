"""Controlled goal progression through self-play and resume."""

import json
from pathlib import Path

from rpbench.core import Backend, load
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation
from rpbench.observations import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"

NPC_TEXT = "Mara Vey is the senior archivist."
GOAL_EVIDENCE = "Mara Vey is the senior archivist."


def test_approved_goal_switches_and_resumes(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": {
                "mode": "mock",
                "id": "goal-test-v1",
                "rules": [],
            },
            "goal_approval_rules": [{
                "goal_id": "subgoal-01",
                "evidence_contains": GOAL_EVIDENCE,
            }],
        },
    }))

    def fake_generate(self, url, payload):
        return NPC_TEXT, {
            "choices": [{
                "message": {"content": NPC_TEXT},
                "finish_reason": "stop",
            }]
        }

    def fake_evaluate(npc_text, settings):
        return {
            "knowledge_updates": [],
            "goal_updates": [{
                "goal_id": "subgoal-01",
                "status": "completed",
                "evidence": GOAL_EVIDENCE,
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)
    monkeypatch.setattr(
        "rpbench.evaluator_runtime.mock_evaluate",
        fake_evaluate,
    )

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    first_state = load(root / "state.json")
    assert first_state["goals"]["subgoal-01"]["status"] == (
        "completed"
    )
    assert first_state["last_confirmed_event"] == 1

    # Replay after resume must preserve the same approved state.
    resume_simulation(root, turns=2)

    state = load(root / "state.json")
    assert state["goals"]["subgoal-01"]["status"] == "completed"
    assert state["last_confirmed_event"] == 1

    requests = [
        e for e in read_jsonl(root / "events.jsonl")
        if e.get("type") == "request"
        and e.get("role") == "player"
    ]
    assert len(requests) == 2

    original_player = load(PLAYER)
    first_goal, second_goal = original_player["subgoals"][:2]

    assert first_goal in requests[0]["payload"]["messages"][0]["content"]
    assert second_goal in requests[1]["payload"]["messages"][0]["content"]

    manifest = load(root / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["completed_turns"] == 2
