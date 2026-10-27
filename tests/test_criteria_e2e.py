"""Live self-play and resume with semantic criterion transitions."""

import json
from pathlib import Path

from rpbench.core import Backend, load
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation
from rpbench.observations import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def test_live_criteria_and_resume(tmp_path, monkeypatch):
    player_prompts = []
    npc_calls = []
    criteria_calls = []

    criteria_definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [
                    {
                        "id": "mara-role",
                        "description": "Learn Mara's responsibility",
                        "evidence_contains": "Senior Archivist",
                    },
                    {
                        "id": "finn-role",
                        "description": "Learn Finn's responsibility",
                        "evidence_contains": "digitization",
                    },
                ],
            }
        },
    }

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "backend": {"name": "mock", "mode": "chat"},
        "runtime": {
            "controlled_player": True,
            "goal_criteria": criteria_definition,
            "semantic_evaluator": {
                "mode": "llm",
                "id": "test-evaluator",
                "backend": {
                    "name": "llamacpp",
                    "base_url": "http://127.0.0.1:18081",
                    "model": "test-evaluator",
                },
                "sampling": {"max_tokens": 2048},
            },
        },
    }))

    def fake_generate(self, url, payload):
        if self.model == "test-evaluator":
            system = payload["messages"][0]["content"]
            context = json.loads(
                payload["messages"][1]["content"]
            )

            if "PLAYER-PERSPECTIVE SCENE BOUNDARIES" in system:
                content = json.dumps({
                    "labels": [
                        {
                            "id": paragraph["id"],
                            "visibility": "observable",
                        }
                        for paragraph in context["paragraphs"]
                    ],
                })
                return content, {
                    "choices": [{
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }],
                }

            if "criterion_updates" in system:
                criteria_calls.append(context)
                count = len(criteria_calls)

                if count == 1:
                    criterion = "mara-role"
                    quote = "I am Mara, Senior Archivist."
                elif count == 2:
                    criterion = "finn-role"
                    quote = "Finn handles digitization."
                else:
                    criterion = None

                changes = []
                if criterion:
                    changes.append({
                        "goal_id": "subgoal-01",
                        "criterion_id": criterion,
                        "status": "satisfied",
                        "evidence": [{
                            "turn": count,
                            "quote": quote,
                        }],
                    })

                content = json.dumps({
                    "schema_version": 1,
                    "criterion_updates": changes,
                })
            else:
                npc_text = context["latest_npc_response"]
                content = json.dumps({
                    "knowledge_updates": [],
                    "goal_updates": [{
                        "goal_id": "subgoal-01",
                        "status": "completed",
                        "evidence": npc_text,
                    }],
                })

            return content, {
                "choices": [{
                    "message": {"content": content},
                    "finish_reason": "stop",
                }],
            }

        system = payload["messages"][0]["content"]

        if "CONTROLLED PLAYER OBJECTIVE:" in system:
            player_prompts.append(system)
            text = "I would like to learn more."
        else:
            npc_calls.append(payload)
            if len(npc_calls) == 1:
                text = "I am Mara, Senior Archivist."
            elif len(npc_calls) == 2:
                text = "Finn handles digitization."
            else:
                text = "The restricted records are in the vault."

        return text, {
            "choices": [{
                "message": {"content": text},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    root = run_simulation(
        SCENARIO, PLAYER, config,
        tmp_path / "runs", turns=2,
    )

    assert load(root / "state.json")["goals"][
        "subgoal-01"
    ]["status"] == "completed"

    assert len(criteria_calls) == 2
    assert len(npc_calls) == 2
    assert len(player_prompts) == 2

    resume_simulation(root, turns=3)

    assert len(criteria_calls) == 2
    assert len(npc_calls) == 3

    assert (
        "Understand why the Blackwood records are restricted."
        in player_prompts[2]
    )

    events = read_jsonl(root / "events.jsonl")
    assert len([
        item for item in events
        if item.get("type") == "semantic_criteria_v1"
    ]) == 2

    assert load(root / "manifest.json")["status"] == "complete"

    # Replaying a completed run must not regenerate evaluations.
    resume_simulation(root, turns=3)
    assert len(criteria_calls) == 2
    assert len(npc_calls) == 3
