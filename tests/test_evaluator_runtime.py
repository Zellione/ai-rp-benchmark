"""Mock evaluation through real self-play and resume."""

import json
from pathlib import Path

from rpbench.core import load
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation
from rpbench.observations import read_jsonl
from rpbench.evaluation_journal import read_evaluations


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def test_mock_evaluation_and_recovery(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": {
                "mode": "mock",
                "id": "test-mock-v1",
                "rules": [{
                    "npc_contains": "Director Vale",
                    "fact": "Director Vale was mentioned.",
                }],
            },
        },
    }))

    def generate(self, url, payload):
        system = payload["messages"][0]["content"]
        answer = (
            "I need more information."
            if "CONTROLLED PLAYER OBJECTIVE:" in system
            else "You should ask Director Vale."
        )
        return answer, {
            "choices": [{
                "message": {"content": answer},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(
        "rpbench.simulation.Backend.generate", generate
    )

    root = run_simulation(
        SCENARIO, PLAYER, config, tmp_path / "runs", turns=1
    )

    knowledge = load(root / "player_knowledge.json")
    assert len(knowledge["facts"]) == 1
    assert knowledge["facts"][0]["status"] == "reported"

    observations = read_jsonl(root / "observations.jsonl")
    events = read_evaluations(
        root / "events.jsonl",
        observations,
        load(root / "state.json"),
    )
    assert len(events) == 1
    assert events[1]["result"]["goal_updates"] == []

    # A missing snapshot must be reconstructed from the journals.
    (root / "player_knowledge.json").unlink()
    resume_simulation(root, turns=2)

    restored = load(root / "player_knowledge.json")
    assert restored["facts"] == knowledge["facts"]

    observations = read_jsonl(root / "observations.jsonl")
    events = read_evaluations(
        root / "events.jsonl",
        observations,
        load(root / "state.json"),
    )

    assert len(events) == 2
    assert load(root / "manifest.json")["status"] == "complete"

    player_requests = [
        event["payload"]["messages"]
        for event in read_jsonl(root / "events.jsonl")
        if event.get("type") == "request"
        and event.get("role") == "player"
    ]
    assert "Director Vale was mentioned." in (
        player_requests[1][0]["content"]
    )
