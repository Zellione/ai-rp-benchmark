"""Ensure live criteria calls respect prerequisite boundaries."""

import json
from pathlib import Path

from rpbench.core import Backend, load
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/blackwood-selfplay.en.json"
PLAYER = ROOT / "players/elena-ward.en.json"


def test_locked_goals_are_not_sent_to_evaluator(tmp_path, monkeypatch):
    criteria_calls = []
    player_prompts = []
    npc_responses = [
        "I am Mara, Senior Archivist.",
        "Finn handles digitization.",
        "The restricted records are in the vault.",
    ]

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [{
                    "id": "mara-role",
                    "description": "Establish Mara's responsibility",
                    "evidence_contains": "Senior Archivist",
                }],
            },
            "subgoal-02": {
                "requires": ["subgoal-01"],
                "criteria": [{
                    "id": "finn-role",
                    "description": "Establish Finn's responsibility",
                    "evidence_contains": "digitization",
                }],
            },
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "backend": {"name": "mock", "mode": "chat"},
            "runtime": {
                "controlled_player": True,
                "goal_criteria": definition,
                "semantic_evaluator": {
                    "mode": "llm",
                    "id": "prerequisite-e2e-evaluator",
                    "quest_scheduler": "direct_supersession",
                    "backend": {
                        "name": "llamacpp",
                        "base_url": "http://127.0.0.1:18081",
                        "model": "test-evaluator",
                    },
                    "sampling": {"max_tokens": 4096},
                },
            },
        }),
        encoding="utf-8",
    )

    def response(content):
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
        }

    def fake_generate(self, url, payload):
        if self.model == "test-evaluator":
            system = payload["messages"][0]["content"]
            context = json.loads(payload["messages"][1]["content"])

            if "PLAYER-PERSPECTIVE SCENE BOUNDARIES" in system:
                return response(json.dumps({
                    "labels": [
                        {"id": p["id"], "visibility": "observable"}
                        for p in context["paragraphs"]
                    ],
                }))

            if "criterion_updates" in system:
                criteria_calls.append(context)
                target = context["active_goal_id"]
                turn = context["latest_npc_response"]["turn"]

                if target == "subgoal-01":
                    assert turn == 1
                    criterion = "mara-role"
                    quote = npc_responses[0]
                elif target == "subgoal-02":
                    assert turn == 2
                    criterion = "finn-role"
                    quote = npc_responses[1]
                else:
                    raise AssertionError(
                        f"Unexpected criteria target: {target}"
                    )

                return response(json.dumps({
                    "schema_version": 1,
                    "criterion_updates": [{
                        "goal_id": target,
                        "criterion_id": criterion,
                        "status": "satisfied",
                        "evidence": [{
                            "turn": turn,
                            "quote": quote,
                        }],
                    }],
                }))

            return response(json.dumps({
                "knowledge_updates": [],
                "goal_updates": [],
            }))

        system = payload["messages"][0]["content"]

        if "CONTROLLED PLAYER OBJECTIVE:" in system:
            player_prompts.append(system)
            return response("I would like to learn more.")

        index = getattr(fake_generate, "npc_index", 0)
        fake_generate.npc_index = index + 1
        return response(npc_responses[index])

    monkeypatch.setattr(Backend, "generate", fake_generate)

    run = run_simulation(
        SCENARIO,
        PLAYER,
        config_path,
        tmp_path / "runs",
        turns=2,
    )

    assert [call["active_goal_id"] for call in criteria_calls] == [
        "subgoal-01",
        "subgoal-02",
    ]

    assert load(run / "state.json")["goals"][
        "subgoal-01"
    ]["status"] == "completed"

    assert load(run / "state.json")["goals"][
        "subgoal-02"
    ]["status"] == "completed"

    assert len(player_prompts) == 2

    # A third turn must not re-adjudicate completed goals.
    resume_simulation(run, turns=3)

    assert len(criteria_calls) == 2
    assert load(run / "manifest.json")["status"] == "complete"
