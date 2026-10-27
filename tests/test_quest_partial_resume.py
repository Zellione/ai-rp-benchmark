"""Resume a turn after opportunity and partial criteria journaling."""

from rpbench import criteria_runtime
from rpbench.core import append, dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl
from rpbench.quest_opportunity_journal import (
    make_opportunity_event,
)
from rpbench.semantic_criteria import make_semantic_event
from rpbench.state import initial_state

from test_quest_opportunity_e2e import (
    DEFINITION,
    GROUND_TRUTH,
)


def test_partial_turn_reuses_original_opportunity_plan(
    tmp_path, monkeypatch,
):
    run = tmp_path / "run"
    run.mkdir()

    dump(run / "ground_truth.json", GROUND_TRUTH)

    observations = [
        {
            "turn": 1,
            "role": "player",
            "text": "Elena asks Voss about the archive.",
        },
        {
            "turn": 1,
            "role": "npc",
            "text": (
                "Voss spontaneously grants Elena "
                "restricted authorization."
            ),
        },
    ]

    candidates = [{
        "goal_id": "subgoal-08",
        "criterion_ids": [
            "authorized-access",
            "document-examined",
        ],
    }]

    selected = [{
        "goal_id": "subgoal-08",
        "criterion_id": "authorized-access",
    }]

    opportunity = make_opportunity_event(
        1, observations, candidates, selected,
        "quest-partial-resume",
        {"usage": {"completion_tokens": 8}},
    )

    approval = make_semantic_event(
        1,
        {
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-08",
                "criterion_id": "authorized-access",
                "status": "satisfied",
                "source_turns": [1],
                "reason": "Explicit restricted authorization.",
            }],
        },
        DEFINITION,
        initial_state(GROUND_TRUTH),
        observations,
        "quest-partial-resume",
        goal_id="subgoal-08",
    )

    append(run / "events.jsonl", opportunity)
    append(run / "events.jsonl", approval)

    calls = []

    def unexpected_screening(*args, **kwargs):
        raise AssertionError(
            "Archived opportunity must be reused"
        )

    def unexpected_criteria(*args, **kwargs):
        calls.append("criteria")
        raise AssertionError(
            "Approval already archived; request superseded"
        )

    monkeypatch.setattr(
        criteria_runtime,
        "check_opportunities",
        unexpected_screening,
    )
    monkeypatch.setattr(
        criteria_runtime,
        "_generate",
        unexpected_criteria,
    )

    settings = {
        "id": "quest-partial-resume",
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

    assert calls == []

    events = read_jsonl(run / "events.jsonl")
    assert len(events) == 2

    # A second resume must be equally deterministic.
    refresh_criteria(
        run, DEFINITION, observations, settings
    )

    assert len(read_jsonl(run / "events.jsonl")) == 2
