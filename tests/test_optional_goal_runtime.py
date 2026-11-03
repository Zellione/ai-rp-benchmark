import json
from copy import deepcopy

import pytest

from rpbench.criteria_runtime import refresh_criteria
from rpbench.goal_criteria import required_goal_summary
from rpbench.state import build_ground_truth, initial_state


def make_fixture(tmp_path):
    player = {
        "name": "Alex",
        "main_goal": "Find and document the key",
        "subgoals": [
            "Optional investigation",
            "Find and secure the key",
            "Complete documented handover",
        ],
    }

    ground_truth = build_ground_truth(player, [])
    state = initial_state(ground_truth)

    (tmp_path / "ground_truth.json").write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )

    definition = {
        "schema_version": 1,
        "goals": {},
    }

    for index in range(1, 4):
        goal_id = f"subgoal-{index:02d}"

        definition["goals"][goal_id] = {
            "requires": (
                [f"subgoal-{index-1:02d}"]
                if index > 1 else []
            ),
            "required": index > 1,
            "evaluate_independently": index > 1,
            "criteria": [{
                "id": "evidence",
                "description": "Confirmed evidence",
                "evidence_contains": "confirmed",
            }],
        }

    return definition, state


def make_progress(definition, completed):
    return {
        "schema_version": 1,
        "goals": {
            goal_id: {
                "proposed_satisfied": goal_id in completed,
                "criteria": {
                    "evidence": {
                        "proposed_satisfied": goal_id in completed,
                        "evidence": [],
                    },
                },
            }
            for goal_id in definition["goals"]
        },
    }


def run_case(tmp_path, monkeypatch, completed):
    definition, state = make_fixture(tmp_path)

    for goal_id in completed:
        state["goals"][goal_id]["status"] = "completed"

    progress = make_progress(definition, completed)

    # The semantic replay itself is tested separately.
    # Here we test the persistent runtime integration.
    def replay(*args, **kwargs):
        return deepcopy(state), deepcopy(progress), []

    monkeypatch.setattr(
        "rpbench.criteria_runtime.advance_criteria_goals",
        replay,
    )

    settings = {
        "id": "integration-test",
        "mode": "llm",
    }

    result_state, result_progress, _ = refresh_criteria(
        tmp_path,
        definition,
        [],
        settings,
    )

    archived = json.loads(
        (tmp_path / "criteria_progress.json").read_text(
            encoding="utf-8"
        )
    )

    expected = required_goal_summary(
        definition, result_state
    )

    assert result_progress["overall_goal"] == expected
    assert archived["overall_goal"] == expected

    # Replaying the same archive must be idempotent.
    first_bytes = (
        tmp_path / "criteria_progress.json"
    ).read_bytes()

    refresh_criteria(
        tmp_path,
        definition,
        [],
        settings,
    )

    second_bytes = (
        tmp_path / "criteria_progress.json"
    ).read_bytes()

    assert second_bytes == first_bytes

    return archived["overall_goal"]


def test_no_required_goals_completed(
    tmp_path, monkeypatch
):
    result = run_case(
        tmp_path, monkeypatch, set()
    )

    assert result["complete"] is False
    assert result["missing_required_goals"] == [
        "subgoal-02",
        "subgoal-03",
    ]


def test_finding_without_handover_is_not_complete(
    tmp_path, monkeypatch
):
    result = run_case(
        tmp_path,
        monkeypatch,
        {"subgoal-02"},
    )

    assert result["complete"] is False
    assert result["missing_required_goals"] == [
        "subgoal-03"
    ]


def test_optional_open_does_not_prevent_completion(
    tmp_path, monkeypatch
):
    result = run_case(
        tmp_path,
        monkeypatch,
        {"subgoal-02", "subgoal-03"},
    )

    assert result["complete"] is True
    assert result["missing_required_goals"] == []
    assert result["optional_goals"] == [
        "subgoal-01"
    ]


def test_runtime_does_not_modify_ground_truth(
    tmp_path, monkeypatch
):
    definition, state = make_fixture(tmp_path)

    before = (
        tmp_path / "ground_truth.json"
    ).read_bytes()

    def replay(*args, **kwargs):
        return (
            deepcopy(state),
            make_progress(definition, set()),
            [],
        )

    monkeypatch.setattr(
        "rpbench.criteria_runtime.advance_criteria_goals",
        replay,
    )

    refresh_criteria(
        tmp_path,
        definition,
        [],
        {"id": "integration-test"},
    )

    assert (
        tmp_path / "ground_truth.json"
    ).read_bytes() == before
