"""Future success must not rewrite earlier quest evaluation state."""

from rpbench import criteria_runtime
from rpbench.core import append, dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl
from rpbench.semantic_criteria import make_semantic_event
from rpbench.state import initial_state


GROUND_TRUTH = {
    "schema_version": 1,
    "player": {
        "subgoals": {
            "subgoal-07": {
                "description": "Request authorization",
                "initial_status": "pending",
            },
            "subgoal-08": {
                "description": "Obtain access and inspect records",
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
                "description": "Submit a restricted-access request",
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
                "description": "Receive official authorization",
                "evidence_contains": "authorization",
            }, {
                "id": "document-examined",
                "description": "Read a restricted document",
                "evidence_contains": "document",
            }],
        },
    },
}


def test_future_archived_approval_does_not_skip_past(
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
                "text": f"Player action in turn {turn}.",
            },
            {
                "turn": turn,
                "role": "npc",
                "text": (
                    "Voss grants official restricted access."
                    if turn == 3
                    else f"NPC response in turn {turn}."
                ),
            },
        ])

    original = initial_state(GROUND_TRUTH)

    # A successful future assessment is already journaled.
    # Older assessments are deliberately missing.
    approval = make_semantic_event(
        3,
        {
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
                "status": "satisfied",
                "source_turns": [3],
                "reason": "Voss explicitly grants access.",
            }],
        },
        DEFINITION,
        original,
        observations[:6],
        "quest-temporal-test",
        goal_id="subgoal-08",
    )

    append(run / "events.jsonl", approval)

    calls = []

    def fake_generate(
        definition, progress, available, settings,
        on_response=None, goal_id=None,
    ):
        turn = available[-1]["turn"]
        calls.append((turn, goal_id))

        updates = [
            {
                "goal_id": goal_id,
                "criterion_id": criterion["id"],
                "status": "not_satisfied",
                "source_turns": [],
                "reason": "Not demonstrated in this turn.",
            }
            for criterion in definition["goals"][goal_id]["criteria"]
            if not progress["goals"][goal_id]["criteria"][
                criterion["id"]
            ]["proposed_satisfied"]
        ]

        return {
            "schema_version": 1,
            "criterion_updates": updates,
        }, {
            "usage": {"completion_tokens": 5},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate
    )

    settings = {
        "id": "quest-temporal-test",
        "max_attempts": 1,
    }

    state, progress, _ = refresh_criteria(
        run, DEFINITION, observations, settings
    )

    request_turns = [
        turn for turn, goal in calls
        if goal == "subgoal-07"
    ]

    # Earlier turns are evaluated normally. The already-archived
    # authorization in turn 3 makes the request unnecessary
    # immediately, so neither turn 3 nor turn 4 evaluates it.
    assert request_turns == [1, 2]

    assert state["goals"]["subgoal-07"]["status"] == "skipped"
    assert state["goals"]["subgoal-08"]["status"] == "pending"

    assert progress["goals"]["subgoal-08"]["criteria"][
        "authorized-access"
    ]["proposed_satisfied"]

    events = read_jsonl(run / "events.jsonl")
    request_events = [
        event for event in events
        if event.get("type") == "semantic_criteria_v1"
        and event.get("goal_id") == "subgoal-07"
    ]

    assert [event["turn"] for event in request_events] == [
        1, 2
    ]

    before = len(calls)

    # Resume must not repeat already archived assessments.
    refresh_criteria(
        run, DEFINITION, observations, settings
    )

    assert len(calls) == before
