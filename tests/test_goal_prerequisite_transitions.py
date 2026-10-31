from copy import deepcopy

import pytest

from rpbench.criteria_progress import advance_criteria_goals
from rpbench.goal_criteria import validate_goal_criteria
from rpbench.player_controller import select_player_goal


def make_fixture():
    state = {
        "schema_version": 1,
        "last_confirmed_event": 0,
        "goals": {
            f"subgoal-{i:02}": {
                "description": f"Objective {i}",
                "status": "pending",
            }
            for i in range(1, 4)
        },
    }

    definition = {
        "schema_version": 1,
        "goals": {},
    }

    for i in range(1, 4):
        goal_id = f"subgoal-{i:02}"
        entry = {
            "criteria": [{
                "id": f"criterion-{i}",
                "description": f"Observable outcome {i}",
                "evidence_contains": f"evidence-{i}",
            }],
        }
        if i > 1:
            entry["requires"] = [f"subgoal-{i-1:02}"]
        definition["goals"][goal_id] = entry

    validate_goal_criteria(definition, state)
    return state, definition


def fake_progress(satisfied):
    return {
        "schema_version": 1,
        "goals": {
            f"subgoal-{i:02}": {
                "proposed_satisfied": i in satisfied,
                "criteria": {
                    f"criterion-{i}": {
                        "proposed_satisfied": i in satisfied,
                        "evidence": [],
                    }
                },
            }
            for i in range(1, 4)
        },
    }


def test_locked_goals_cannot_complete(monkeypatch):
    state, definition = make_fixture()
    original = deepcopy(state)

    monkeypatch.setattr(
        "rpbench.criteria_progress.replay_semantic_criteria",
        lambda *args: fake_progress({2, 3}),
    )

    updated, progress, transitions = advance_criteria_goals(
        state, definition, [], []
    )

    assert updated["goals"]["subgoal-01"]["status"] == "pending"
    assert updated["goals"]["subgoal-02"]["status"] == "pending"
    assert updated["goals"]["subgoal-03"]["status"] == "pending"
    assert transitions == []
    assert state == original


def test_completed_predecessor_unlocks_next(monkeypatch):
    state, definition = make_fixture()
    state["goals"]["subgoal-01"]["status"] = "completed"

    monkeypatch.setattr(
        "rpbench.criteria_progress.replay_semantic_criteria",
        lambda *args: fake_progress({2, 3}),
    )

    updated, _, transitions = advance_criteria_goals(
        state, definition, [], []
    )

    assert updated["goals"]["subgoal-02"]["status"] == "completed"
    assert updated["goals"]["subgoal-03"]["status"] == "completed"
    assert len(transitions) == 2


def test_first_goal_is_selected_before_unlock(monkeypatch):
    state, definition = make_fixture()

    assert select_player_goal(state, definition)["id"] == "subgoal-01"

    state["goals"]["subgoal-01"]["status"] = "completed"

    assert select_player_goal(state, definition)["id"] == "subgoal-02"
