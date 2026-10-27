"""Fail-open scheduling and opportunity journal integrity."""

import json

import pytest

from rpbench import criteria_runtime
from rpbench.core import append, dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl
from rpbench.quest_opportunity_journal import (
    make_opportunity_event,
    read_opportunity_event,
)

from test_quest_opportunity_e2e import (
    GROUND_TRUTH,
    DEFINITION,
)


def observations_for_turn(text="Mara answers Elena."):
    return [
        {
            "turn": 1,
            "role": "player",
            "text": "Elena asks Mara about access.",
        },
        {
            "turn": 1,
            "role": "npc",
            "text": text,
        },
    ]


CANDIDATES = [{
    "goal_id": "subgoal-08",
    "criterion_ids": [
        "authorized-access",
        "document-examined",
    ],
}]


def test_opportunity_failure_uses_full_evaluation_and_replays(
    tmp_path, monkeypatch,
):
    dump(tmp_path / "ground_truth.json", GROUND_TRUTH)
    observations = observations_for_turn()

    checks = []
    assessed = []

    def failed_opportunity(
        definition, candidates, available, settings,
    ):
        checks.append(available[-1]["turn"])
        raise ValueError("Invalid opportunity JSON")

    def fake_generate(
        definition, progress, available, settings,
        on_response=None, goal_id=None,
    ):
        assessed.append(goal_id)
        updates = [
            {
                "goal_id": goal_id,
                "criterion_id": criterion["id"],
                "status": "not_satisfied",
                "source_turns": [],
                "reason": "Not demonstrated",
            }
            for criterion in definition["goals"][goal_id]["criteria"]
        ]
        return {
            "schema_version": 1,
            "criterion_updates": updates,
        }, {
            "usage": {"completion_tokens": 5},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        criteria_runtime, "check_opportunities",
        failed_opportunity,
    )
    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate,
    )

    settings = {
        "id": "quest-safety",
        "max_attempts": 1,
        "quest_opportunity_enabled": True,
    }

    refresh_criteria(
        tmp_path, DEFINITION, observations, settings
    )

    assert checks == [1]
    assert assessed == ["subgoal-07", "subgoal-08"]

    events = read_jsonl(tmp_path / "events.jsonl")
    fallback_events = [
        event for event in events
        if event["type"] == "quest_opportunity_v1"
    ]

    assert len(fallback_events) == 1
    assert fallback_events[0]["metadata"]["fallback"] is True

    counts = (len(checks), len(assessed), len(events))

    refresh_criteria(
        tmp_path, DEFINITION, observations, settings
    )

    assert (
        len(checks),
        len(assessed),
        len(read_jsonl(tmp_path / "events.jsonl")),
    ) == counts


def test_opportunity_journal_rejects_source_mutation():
    observations = observations_for_turn()

    event = make_opportunity_event(
        1, observations, CANDIDATES, [],
        "quest-safety",
    )

    changed = observations_for_turn(
        "Mara secretly authorizes Elena."
    )

    with pytest.raises(
        ValueError, match="provenance mismatch"
    ):
        read_opportunity_event(
            [event], 1, changed, CANDIDATES,
            "quest-safety",
        )


def test_opportunity_journal_rejects_candidate_mutation():
    observations = observations_for_turn()

    event = make_opportunity_event(
        1, observations, CANDIDATES, [],
        "quest-safety",
    )

    changed = [{
        "goal_id": "subgoal-08",
        "criterion_ids": ["document-examined"],
    }]

    with pytest.raises(
        ValueError, match="provenance mismatch"
    ):
        read_opportunity_event(
            [event], 1, observations, changed,
            "quest-safety",
        )


def test_opportunity_journal_rejects_duplicate_events():
    observations = observations_for_turn()
    event = make_opportunity_event(
        1, observations, CANDIDATES, [],
        "quest-safety",
    )

    with pytest.raises(
        ValueError, match="Duplicate opportunity event"
    ):
        read_opportunity_event(
            [event, event], 1, observations,
            CANDIDATES, "quest-safety",
        )
