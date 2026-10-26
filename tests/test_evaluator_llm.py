"""LLM evaluator, self-play, and archived-resume integration."""

import json
from pathlib import Path

import pytest

from rpbench.core import Backend, load
from rpbench.evaluator_llm import evaluate_llm
from rpbench.player_evaluator import validate_evaluation
from rpbench.evaluator_runtime import validate_evaluator_config
from rpbench.observations import read_jsonl
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"
NPC_TEXT = "Director Vale manages the restricted archive."


def evaluator_config():
    return {
        "mode": "llm",
        "id": "test-evaluator-v1",
        "backend": {
            "name": "llamacpp",
            "base_url": "http://127.0.0.1:18081",
            "model": "evaluator-test",
        },
        "sampling": {
            "temperature": 0,
            "max_tokens": 512,
        },
    }


def test_valid_llm_response(monkeypatch):
    def fake_generate(self, url, payload):
        assert self.model == "evaluator-test"
        assert url.endswith("/v1/chat/completions")

        return json.dumps({
            "knowledge_updates": [{
                "fact": "Director Vale manages the archive.",
                "evidence": NPC_TEXT,
            }],
            "goal_updates": [],
        }), {}

    monkeypatch.setattr(Backend, "generate", fake_generate)

    result = evaluate_llm(
        NPC_TEXT,
        {"goals": {}},
        {"facts": []},
        evaluator_config(),
    )

    assert result["knowledge_updates"][0]["fact"] == (
        "Director Vale manages the archive."
    )
    assert "status" not in result["knowledge_updates"][0]


def test_invalid_json_rejected(monkeypatch):
    monkeypatch.setattr(
        Backend, "generate",
        lambda self, url, payload: ("not JSON", {}),
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        evaluate_llm(
            NPC_TEXT,
            {"goals": {}},
            {"facts": []},
            evaluator_config(),
        )


def test_invalid_evidence_rejected(monkeypatch):
    monkeypatch.setattr(
        Backend, "generate",
        lambda self, url, payload: (
            json.dumps({
                "knowledge_updates": [{
                    "fact": "Unsupported claim",
                    "evidence": "Never spoken",
                }],
                "goal_updates": [],
            }),
            {},
        ),
    )

    proposal = evaluate_llm(
        NPC_TEXT,
        {"goals": {}},
        {"facts": []},
        evaluator_config(),
    )

    with pytest.raises(ValueError, match="evidence"):
        validate_evaluation(
            proposal,
            NPC_TEXT,
            {"goals": {}},
        )


def test_invalid_backend_configuration():
    settings = evaluator_config()
    settings["backend"]["name"] = "unknown"

    with pytest.raises(ValueError):
        validate_evaluator_config(settings)


def test_selfplay_llm_evaluator_resume(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "mode": "chat",
        },
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": evaluator_config(),
        },
    }))

    calls = {"evaluator": 0}

    def fake_generate(self, url, payload):
        if self.model == "evaluator-test":
            calls["evaluator"] += 1
            content = json.loads(
                payload["messages"][1]["content"]
            )
            response = content["latest_npc_response"]

            return json.dumps({
                "knowledge_updates": [{
                    "fact": "Director Vale manages the archive.",
                    "evidence": response,
                }],
                "goal_updates": [],
            }), {}

        messages = payload["messages"]
        is_player = (
            "CONTROLLED PLAYER OBJECTIVE:"
            in messages[0]["content"]
        )
        answer = "Who manages the archive?" if is_player else NPC_TEXT
        return answer, {
            "choices": [{
                "message": {"content": answer},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=1,
    )

    assert calls["evaluator"] == 1

    knowledge = load(root / "player_knowledge.json")
    assert len(knowledge["facts"]) == 1

    resume_simulation(root, turns=2)
    assert calls["evaluator"] == 2

    events = read_jsonl(root / "events.jsonl")
    evaluations = [
        e for e in events
        if e.get("type") == "player_evaluation_v1"
    ]
    assert len(evaluations) == 2

    player_requests = [
        e["payload"]["messages"]
        for e in events
        if e.get("type") == "request"
        and e.get("role") == "player"
    ]
    assert "Director Vale manages the archive." in (
        player_requests[1][0]["content"]
    )

    # Existing complete turns must never be evaluated again.
    resume_simulation(root, turns=2)
    assert calls["evaluator"] == 2


def test_invalid_llm_json_leaves_no_evaluation_event(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": evaluator_config(),
        },
    }))

    def fake_generate(self, url, payload):
        if self.model == "evaluator-test":
            return "invalid JSON", {}

        answer = NPC_TEXT
        return answer, {
            "choices": [{
                "message": {"content": answer},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    with pytest.raises(ValueError, match="invalid JSON"):
        run_simulation(
            SCENARIO, PLAYER, config,
            tmp_path / "runs", turns=1,
        )

    roots = list((tmp_path / "runs").iterdir())
    assert len(roots) == 1

    events = read_jsonl(roots[0] / "events.jsonl")
    assert not any(
        event.get("type") == "player_evaluation_v1"
        for event in events
    )

    assert any(
        event.get("type") == "response"
        and event.get("role") == "npc"
        and event.get("response_status") == "complete"
        for event in events
    )


def test_resume_evaluates_archived_response_once(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "semantic_evaluator": evaluator_config(),
        },
    }))

    calls = {"evaluator": 0}

    def fake_generate(self, url, payload):
        if self.model == "evaluator-test":
            calls["evaluator"] += 1

            if calls["evaluator"] == 1:
                return "invalid JSON", {}

            return json.dumps({
                "knowledge_updates": [{
                    "fact": "Director Vale manages the archive.",
                    "evidence": NPC_TEXT,
                }],
                "goal_updates": [],
            }), {}

        return NPC_TEXT, {
            "choices": [{
                "message": {"content": NPC_TEXT},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    with pytest.raises(ValueError, match="invalid JSON"):
        run_simulation(
            SCENARIO, PLAYER, config,
            tmp_path / "runs", turns=1,
        )

    root = next((tmp_path / "runs").iterdir())

    # Resume should recover the completed NPC response from
    # events.jsonl and evaluate it, without another NPC call.
    resume_simulation(root, turns=1)

    assert calls["evaluator"] == 2

    events = read_jsonl(root / "events.jsonl")
    evaluations = [
        event for event in events
        if event.get("type") == "player_evaluation_v1"
    ]

    assert len(evaluations) == 1
    assert load(root / "manifest.json")["status"] == "complete"


def test_knowledge_only_ignores_goal_proposals(monkeypatch):
    def fake_generate(self, url, payload):
        system = payload["messages"][0]["content"]
        assert "KNOWLEDGE-ONLY MODE" in system

        text = json.dumps({
            "knowledge_updates": [{
                "fact": "Director Vale manages the restricted archive.",
                "evidence": NPC_TEXT,
            }],
            "goal_updates": [{
                "goal_id": "subgoal-01",
                "status": "completed",
                "evidence": "Fabricated quote",
            }],
        })
        return text, {
            "choices": [{
                "message": {"content": text},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    result = evaluate_llm(
        NPC_TEXT,
        {"goals": {
            "subgoal-01": {
                "status": "pending",
                "description": "Find staff",
            },
        }},
        {"facts": []},
        evaluator_config(),
        knowledge_only=True,
    )

    assert result["goal_updates"] == []
    assert len(result["knowledge_updates"]) == 1
    assert result["knowledge_updates"][0]["evidence"] == NPC_TEXT
