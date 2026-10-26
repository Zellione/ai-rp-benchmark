import json
from pathlib import Path

from rpbench.core import Backend, load
from rpbench.observations import read_jsonl
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def test_retry_rejects_bad_evidence_and_preserves_npc(
    tmp_path, monkeypatch,
):
    calls = {"npc": 0, "evaluator": 0}

    def generate(self, url, payload):
        if self.model == "evaluator-test":
            calls["evaluator"] += 1

            if calls["evaluator"] == 1:
                evidence = "Mara ... Archivist"
            else:
                evidence = "Mara is the Senior Archivist."

            text = json.dumps({
                "knowledge_updates": [{
                    "fact": "Mara is the Senior Archivist.",
                    "evidence": evidence,
                }],
                "goal_updates": [],
            })

            return text, {
                "choices": [{
                    "message": {
                        "content": text,
                        "reasoning_content": "Reasoning",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 100},
            }

        is_player = (
            "CONTROLLED PLAYER OBJECTIVE:"
            in payload["messages"][0]["content"]
        )

        if not is_player:
            calls["npc"] += 1

        content = (
            "Who manages the archive?"
            if is_player
            else "Mara is the Senior Archivist."
        )

        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": {
                "mode": "llm",
                "id": "retry-test",
                "backend": {
                    "name": "llamacpp",
                    "base_url": "http://127.0.0.1:18081",
                    "model": "evaluator-test",
                },
                "sampling": {"max_tokens": 256},
                "max_attempts": 2,
            },
        },
    }))

    run = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    assert calls == {"npc": 1, "evaluator": 2}

    events = read_jsonl(run / "events.jsonl")
    attempts = [
        item for item in events
        if item.get("type") == "player_evaluation_attempt_v1"
    ]
    assert [item["status"] for item in attempts] == [
        "rejected", "accepted",
    ]
    assert "Mara ... Archivist" in (
        attempts[0]["raw_response"]["content"]
    )
    assert attempts[0]["raw_response"]["reasoning"] == "Reasoning"

    evaluations = [
        item for item in events
        if item.get("type") == "player_evaluation_v1"
    ]
    assert len(evaluations) == 1

    knowledge = load(run / "player_knowledge.json")
    assert len(knowledge["facts"]) == 1

    resume_simulation(run, turns=1)
    assert calls == {"npc": 1, "evaluator": 2}


def test_resume_overrides_evaluator_budget(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": {
                "mode": "llm",
                "id": "budget-test",
                "backend": {
                    "name": "llamacpp",
                    "base_url": "http://127.0.0.1:18081",
                    "model": "evaluator-test",
                },
                "sampling": {"max_tokens": 256},
            },
        },
    }))

    def generate(self, url, payload):
        if self.model == "evaluator-test":
            if payload["max_tokens"] == 256:
                return "", {
                    "choices": [{
                        "message": {
                            "content": "",
                            "reasoning_content": "Thinking",
                        },
                        "finish_reason": "length",
                    }],
                }

            text = json.dumps({
                "knowledge_updates": [],
                "goal_updates": [],
            })
            return text, {
                "choices": [{
                    "message": {"content": text},
                    "finish_reason": "stop",
                }],
            }

        text = "Mara is the Senior Archivist."
        return text, {
            "choices": [{
                "message": {"content": text},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", generate)

    import pytest

    with pytest.raises(ValueError, match="invalid JSON"):
        run_simulation(
            SCENARIO, PLAYER, config,
            tmp_path / "runs", turns=1,
        )

    run = next((tmp_path / "runs").iterdir())

    resume_simulation(
        run,
        turns=1,
        evaluator_max_tokens=8192,
        evaluator_attempts=2,
    )

    manifest = load(run / "manifest.json")
    settings = manifest["runtime"]["semantic_evaluator"]

    assert manifest["status"] == "complete"
    assert settings["sampling"]["max_tokens"] == 8192
    assert settings["max_attempts"] == 2

    events = read_jsonl(run / "events.jsonl")
    assert any(
        e.get("type") == "evaluator_config_override_v1"
        for e in events
    )
    assert len([
        e for e in events
        if e.get("type") == "player_evaluation_v1"
    ]) == 1


def test_evaluator_json_fences():
    import pytest
    from rpbench.evaluator_llm import parse_evaluator_json

    expected = {
        "knowledge_updates": [],
        "goal_updates": [],
    }
    body = json.dumps(expected, indent=2)

    assert parse_evaluator_json(body) == expected
    assert parse_evaluator_json(
        "```json\n" + body + "\n```"
    ) == expected
    assert parse_evaluator_json(
        "```\n" + body + "\n```"
    ) == expected

    for invalid in (
        "Here is the result:\n" + body,
        "```json\n" + body + "\n```\nExplanation",
        "```json\n" + body,
        body + "\nAdditional commentary",
    ):
        with pytest.raises(json.JSONDecodeError):
            parse_evaluator_json(invalid)
