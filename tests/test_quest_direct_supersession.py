"""Direct supersession without opportunity screening."""

from rpbench import criteria_runtime
from rpbench.core import dump
from rpbench.criteria_runtime import refresh_criteria
from rpbench.observations import read_jsonl

from test_quest_opportunity_e2e import (
    DEFINITION,
    GROUND_TRUTH,
)


SETTINGS = {
    "id": "direct-supersession-test",
    "max_attempts": 1,
    "quest_scheduler": "direct_supersession",
}


def setup_run(root, *, authorized):
    dump(root / "ground_truth.json", GROUND_TRUTH)

    observations = []
    for turn in (1, 2):
        observations.extend([
            {
                "turn": turn,
                "role": "player",
                "text": f"Elena asks about access in turn {turn}.",
            },
            {
                "turn": turn,
                "role": "npc",
                "text": (
                    "Voss authorizes access immediately."
                    if authorized and turn == 1
                    else f"Conversation in turn {turn}."
                ),
            },
        ])
    return observations


def install_model(monkeypatch, calls, *, authorized):
    def fake_generate(
        definition, progress, available, settings,
        on_response=None, goal_id=None, criterion_ids=None,
    ):
        turn = available[-1]["turn"]
        calls.append((turn, goal_id, criterion_ids))

        ids = (
            criterion_ids
            if criterion_ids is not None
            else [
                c["id"]
                for c in definition["goals"][goal_id]["criteria"]
                if not progress["goals"][goal_id]["criteria"][
                    c["id"]
                ]["proposed_satisfied"]
            ]
        )

        updates = []
        for criterion_id in ids:
            satisfied = (
                authorized
                and turn == 1
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
                    "Explicit permission"
                    if satisfied else "Not demonstrated"
                ),
            })

        return {
            "schema_version": 1,
            "criterion_updates": updates,
        }, {
            "usage": {"completion_tokens": 10},
            "finish_reason": "stop",
        }

    def fail_opportunity(*args, **kwargs):
        raise AssertionError(
            "Direct scheduling must not run opportunity screening"
        )

    monkeypatch.setattr(
        criteria_runtime, "_generate", fake_generate
    )
    monkeypatch.setattr(
        criteria_runtime, "check_opportunities", fail_opportunity
    )


def test_direct_approval_skips_request_and_resumes(
    tmp_path, monkeypatch,
):
    observations = setup_run(tmp_path, authorized=True)
    calls = []
    install_model(monkeypatch, calls, authorized=True)

    state, progress, _ = refresh_criteria(
        tmp_path, DEFINITION, observations, SETTINGS
    )

    assert state["goals"]["subgoal-07"]["status"] == "skipped"
    assert state["goals"]["subgoal-08"]["status"] == "pending"

    assert calls[0] == (
        1, "subgoal-08", ["authorized-access"]
    )
    assert (1, "subgoal-07", None) not in calls
    assert (2, "subgoal-07", None) not in calls

    assert progress["goals"]["subgoal-08"]["criteria"][
        "authorized-access"
    ]["proposed_satisfied"]

    assert not progress["goals"]["subgoal-08"]["criteria"][
        "document-examined"
    ]["proposed_satisfied"]

    events = read_jsonl(tmp_path / "events.jsonl")
    assert not any(
        e["type"] == "quest_opportunity_v1"
        for e in events
    )

    before = (len(calls), len(events))

    refresh_criteria(
        tmp_path, DEFINITION, observations, SETTINGS
    )

    assert (
        len(calls),
        len(read_jsonl(tmp_path / "events.jsonl")),
    ) == before


def test_direct_denial_keeps_request_active(
    tmp_path, monkeypatch,
):
    observations = setup_run(tmp_path, authorized=False)
    calls = []
    install_model(monkeypatch, calls, authorized=False)

    state, _, _ = refresh_criteria(
        tmp_path, DEFINITION, observations, SETTINGS
    )

    assert state["goals"]["subgoal-07"]["status"] == "pending"

    assert calls == [
        (1, "subgoal-08", ["authorized-access"]),
        (1, "subgoal-07", None),
        (2, "subgoal-08", ["authorized-access"]),
        (2, "subgoal-07", None),
    ]
