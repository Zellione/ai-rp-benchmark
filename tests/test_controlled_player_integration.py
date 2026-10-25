"""Controlled player integration with self-play and resume."""

import json
from pathlib import Path

import pytest

from rpbench.simulation import run_simulation
from rpbench.resume import resume_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def write_config(tmp_path, controlled):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "mode": "chat",
            "model": "mock",
        },
        "sampling": {},
        "runtime": {"controlled_player": controlled},
    }))
    return path


def requests(root, role):
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text().splitlines()
    ]
    return [
        event["payload"]["messages"]
        for event in events
        if event.get("type") == "request"
        and event.get("role") == role
    ]


def test_controlled_player_survives_resume(tmp_path):
    config = write_config(tmp_path, True)

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    first_player = requests(root, "player")[0]
    assert "CONTROLLED PLAYER OBJECTIVE:" in (
        first_player[0]["content"]
    )

    player = json.loads(PLAYER.read_text(encoding="utf-8"))
    first_goal = player["subgoals"][0]

    assert first_goal in first_player[0]["content"]

    first_npc = requests(root, "npc")[0]
    assert "CONTROLLED PLAYER OBJECTIVE:" not in (
        first_npc[0]["content"]
    )

    resume_simulation(root, turns=2)

    player_requests = requests(root, "player")
    assert len(player_requests) == 2

    assert player_requests[1][0] == player_requests[0][0]

    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    assert manifest["status"] == "complete"
    assert manifest["completed_turns"] == 2
    assert manifest["runtime"]["controlled_player"] is True


def test_legacy_player_remains_unchanged(tmp_path):
    config = write_config(tmp_path, False)

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    player_messages = requests(root, "player")[0]

    assert "CONTROLLED PLAYER OBJECTIVE:" not in (
        player_messages[0]["content"]
    )


@pytest.mark.parametrize("invalid", ["yes", 1, None])
def test_invalid_control_option_rejected_before_run(
    tmp_path, invalid
):
    config = write_config(tmp_path, invalid)

    with pytest.raises(ValueError, match="controlled_player"):
        run_simulation(
            SCENARIO, PLAYER, config,
            tmp_path / "runs", turns=1,
        )

    assert not (tmp_path / "runs").exists()
