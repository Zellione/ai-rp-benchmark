import json
from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.judge_calibration import load_suite
from rpbench.offline_goal_audit import (
    SCHEMA_VERSION,
    validate_verdict,
)

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = ROOT / (
    "tests/fixtures/quest_goal04_v4_hard_ground_truth.json"
)

DEFINITION = ROOT / (
    "experiments/simulation/quest-archivist-v1/"
    "mistral-glm-flash-player-v4-en.json"
)

suite, definition = load_suite(FIXTURE, DEFINITION)


@pytest.mark.parametrize(
    "case",
    suite["cases"],
    ids=lambda item: item["id"],
)
def test_hard_ground_truth_is_structurally_valid(case):
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "goal_id": "subgoal-04",
        "criteria": [
            {
                "criterion_id": cid,
                **deepcopy(case["expected"][cid]),
            }
            for cid in ("entry", "details")
        ],
    }

    validated = validate_verdict(
        verdict,
        "subgoal-04",
        definition,
        case["observations"],
    )

    assert len(validated["criteria"]) == 2


def test_hard_case_ids_are_unique():
    ids = [item["id"] for item in suite["cases"]]
    assert len(ids) == len(set(ids)) == 6


def test_hard_cases_have_ordered_turns():
    for case in suite["cases"]:
        turns = [item["turn"] for item in case["observations"]]
        assert turns == sorted(turns)
        assert turns[0] == 1
        assert turns[-1] >= 3


def test_hard_cases_have_expected_results():
    expected = {
        "delayed-disclosure": (2, 3),
        "corrected-entry": (3, 3),
        "player-assumption": (None, None),
        "duplicate-records": (3, 3),
        "explicit-denial": (None, None),
        "late-independent-entry": (5, 5),
    }

    for case in suite["cases"]:
        entry = case["expected"]["entry"]["first_satisfied_turn"]
        details = case["expected"]["details"]["first_satisfied_turn"]
        assert (entry, details) == expected[case["id"]]


def test_hard_cases_do_not_mutate_baseline():
    baseline = json.loads((
        ROOT / "tests/fixtures/quest_goal04_v4_ground_truth.json"
    ).read_text(encoding="utf-8"))

    hard_ids = {item["id"] for item in suite["cases"]}
    baseline_ids = {item["id"] for item in baseline["cases"]}

    assert not hard_ids.intersection(baseline_ids)
