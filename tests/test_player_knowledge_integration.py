"""Player knowledge persistence and resume integration."""

import json
from pathlib import Path

from rpbench.core import load
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"

RULE = {
    "id": "director-vale",
    "npc_contains": "Director Vale",
    "fact": "The NPC mentioned Director Vale.",
}


def setup_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "mode": "chat",
            "model": "mock",
        },
        "sampling": {},
        "runtime": {
            "controlled_player": True,
            "player_knowledge_rules": [RULE],
        },
    }))
    return path


def player_requests(root):
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text().splitlines()
    ]
    return [
        e["payload"]["messages"]
        for e in events
        if e.get("type") == "request"
        and e.get("role") == "player"
    ]


def test_knowledge_is_injected_and_survives_resume(
    tmp_path, monkeypatch
):
    config = setup_config(tmp_path)

    def fake_generate(self, url, payload):
        messages = payload["messages"]
        system = messages[0]["content"]

        if "CONTROLLED PLAYER OBJECTIVE:" in system:
            answer = "Who manages these records?"
        else:
            answer = "You should speak with Director Vale."

        return answer, {
            "choices": [{
                "message": {"content": answer},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(
        "rpbench.simulation.Backend.generate",
        fake_generate,
    )

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    knowledge = load(root / "player_knowledge.json")
    assert len(knowledge["facts"]) == 1
    assert knowledge["facts"][0]["status"] == "reported"
    assert knowledge["facts"][0]["turn"] == 1

    first = player_requests(root)[0][0]["content"]
    assert "No tracked facts" in first

    # Force reconstruction from the observation archive.
    (root / "player_knowledge.json").unlink()

    resume_simulation(root, turns=2)

    second = player_requests(root)[1][0]["content"]
    assert "[REPORTED, turn 1]" in second
    assert "Director Vale" in second

    restored = load(root / "player_knowledge.json")
    assert len(restored["facts"]) == 1
    assert restored["facts"][0]["turn"] == 1

    manifest = load(root / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["completed_turns"] == 2


def test_no_knowledge_leaks_into_npc_prompt(tmp_path, monkeypatch):
    config = setup_config(tmp_path)

    captured = []

    def fake_generate(self, url, payload):
        captured.append(payload["messages"])
        return "Director Vale can help.", {
            "choices": [{
                "message": {"content": "Director Vale can help."},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(
        "rpbench.simulation.Backend.generate",
        fake_generate,
    )

    run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=2,
    )

    npc_requests = captured[1::2]

    for messages in npc_requests:
        assert "PLAYER KNOWLEDGE:" not in json.dumps(messages)


def test_resume_recovers_stale_knowledge_snapshot(
    tmp_path, monkeypatch
):
    """Archived NPC dialogue wins over a stale knowledge snapshot."""
    config = setup_config(tmp_path)

    def fake_generate(self, url, payload):
        system = payload["messages"][0]["content"]

        if "CONTROLLED PLAYER OBJECTIVE:" in system:
            answer = "Who is in charge?"
        else:
            answer = "Ask Director Vale."

        return answer, {
            "choices": [{
                "message": {"content": answer},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(
        "rpbench.simulation.Backend.generate",
        fake_generate,
    )

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    # Simulate a crash before the knowledge snapshot update.
    stale = {
        "schema_version": 1,
        "facts": [],
    }
    from rpbench.core import dump
    dump(root / "player_knowledge.json", stale)

    resume_simulation(root, turns=3)

    knowledge = load(root / "player_knowledge.json")

    assert len(knowledge["facts"]) == 1
    assert knowledge["facts"][0]["id"] == "director-vale"
    assert knowledge["facts"][0]["turn"] == 1

    prompts = player_requests(root)
    assert len(prompts) == 3

    for prompt in prompts[1:]:
        system = prompt[0]["content"]
        assert system.count("[REPORTED, turn 1]") == 1


def test_knowledge_replay_matches_incremental_updates():
    from rpbench.player_knowledge import (
        initial_player_knowledge,
        replay_player_knowledge,
        update_player_knowledge,
    )

    player = {"known_facts": ["The archive exists."]}

    observations = [
        {"turn": 1, "role": "player", "text": "Hello"},
        {
            "turn": 1,
            "role": "npc",
            "text": "Ask Director Vale.",
        },
        {"turn": 2, "role": "player", "text": "Why?"},
        {
            "turn": 2,
            "role": "npc",
            "text": "Director Vale handles this.",
        },
    ]

    incremental = initial_player_knowledge(player)

    for entry in observations:
        if entry["role"] == "npc":
            incremental = update_player_knowledge(
                incremental,
                entry["turn"],
                entry["text"],
                [RULE],
            )

    replayed = replay_player_knowledge(
        player, observations, [RULE]
    )

    assert incremental == replayed
    assert len(replayed["facts"]) == 2
