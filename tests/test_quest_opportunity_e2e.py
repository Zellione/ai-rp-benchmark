"""Integrated quest opportunity screening and resume."""

from rpbench import criteria_runtime
from rpbench.core import dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl


GROUND_TRUTH = {
    "schema_version": 1,
    "player": {
        "subgoals": {
            "subgoal-07": {
                "description": "Request authorization",
                "initial_status": "pending",
            },
            "subgoal-08": {
                "description": "Obtain access and examine documents",
                "initial_status": "pending",
            },
        },
    },
    "characters": {},
}

DEFINITION = {
    "schema_version": 1,
    "goals": {
        "subgoal-07": {
            "criteria": [{
                "id": "authorization-request",
                "description": "Submit an authorization request",
                "evidence_contains": "request",
            }],
            "superseded_by": {
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
            },
        },
        "subgoal-08": {
            "criteria": [{
                "id": "authorized-access",
                "description": "Receive valid restricted authorization",
                "evidence_contains": "access",
            }, {
                "id": "document-examined",
                "description": "Examine a restricted document",
                "evidence_contains": "document",
            }],
        },
    },
}


def test_spontaneous_access_skips_request_and_resumes(
    tmp_path, monkeypatch,
):
    run = tmp_path / "run"
    run.mkdir()
    dump(run / "ground_truth.json", GROUND_TRUTH)

    observations = []
    for turn in range(1, 5):
        observations.extend([
            {
                "turn": turn,
                "role": "player",
                "text": f"Elena acts in turn {turn}.",
            },
            {
                "turn": turn,
                "role": "npc",
                "text": (
                    "Voss grants restricted authorization."
                    if turn == 3
                    else f"NPC responds in turn {turn}."
                ),
            },
        ])

    screening_calls = []
    criteria_calls = []

    def fake_opportunity(
        definition, candidates, available, settings,
    ):
        turn = available[-1]["turn"]
        screening_calls.append(turn)

        if turn == 3:
            return [{
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
            }], {
                "usage": {"completion_tokens": 10},
                "finish_reason": "stop",
            }

        return [], {
            "usage": {"completion_tokens": 5},
            "finish_reason": "stop",
        }

    def fake_generate(
        definition, progress, available, settings,
        on_response=None, goal_id=None,
    ):
        turn = available[-1]["turn"]
        criteria_calls.append((turn, goal_id))

        updates = []
        for criterion in definition["goals"][goal_id]["criteria"]:
            criterion_id = criterion["id"]

            if progress["goals"][goal_id]["criteria"][
                criterion_id
            ]["proposed_satisfied"]:
                continue

            satisfied = (
                turn == 3
                and goal_id == "subgoal-08"
                and criterion_id == "authorized-access"
            )

            updates.append({
                "goal_id": goal_id,
                "criterion_id": criterion_id,
                "status": (
                    "satisfied" if satisfied else "not_satisfied"
                ),
                "source_turns": [turn] if satisfied else [],
                "reason": (
                    "Explicit approval" if satisfied
                    else "No evidence in this turn"
                ),
            })

        return {
            "schema_version": 1,
            "criterion_updates": updates,
        }, {
            "usage": {"completion_tokens": 20},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        criteria_runtime,
        "check_opportunities",
        fake_opportunity,
    )
    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate
    )

    settings = {
        "id": "quest-e2e",
        "max_attempts": 1,
        "quest_opportunity_enabled": True,
    }

    state, progress, _ = refresh_criteria(
        run, DEFINITION, observations, settings
    )

    assert state["goals"]["subgoal-07"]["status"] == "skipped"
    assert state["goals"]["subgoal-08"]["status"] == "pending"

    assert progress["goals"]["subgoal-08"]["criteria"][
        "authorized-access"
    ]["proposed_satisfied"]

    # The opportunity is checked BEFORE the request goal.
    # A confirmed approval makes the request unnecessary.
    assert (3, "subgoal-08") in criteria_calls
    assert (3, "subgoal-07") not in criteria_calls
    assert (4, "subgoal-07") not in criteria_calls

    events = read_jsonl(run / "events.jsonl")
    opportunities = [
        event for event in events
        if event.get("type") == "quest_opportunity_v1"
    ]
    assert opportunities
    assert any(
        event["turn"] == 3
        and event["selected"] == [{
            "goal_id": "subgoal-08",
            "criterion_id": "authorized-access",
        }]
        for event in opportunities
    )

    previous = (
        len(screening_calls),
        len(criteria_calls),
        len(events),
    )

    refresh_criteria(
        run, DEFINITION, observations, settings
    )

    assert (
        len(screening_calls),
        len(criteria_calls),
        len(read_jsonl(run / "events.jsonl")),
    ) == previous
