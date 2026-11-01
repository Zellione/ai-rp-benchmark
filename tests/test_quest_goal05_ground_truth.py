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
    "tests/fixtures/quest_goal05_v4_ground_truth.json"
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
def test_ground_truth_valid(case):
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "goal_id": "subgoal-05",
        "criteria": [
            {
                "criterion_id": cid,
                **deepcopy(case["expected"][cid]),
            }
            for cid in ("name", "corroboration")
        ],
    }

    result = validate_verdict(
        verdict,
        "subgoal-05",
        definition,
        case["observations"],
    )

    assert len(result["criteria"]) == 2


def test_expected_status_and_turns():
    expected = {
        "anonymous-designation": (
            "partial", None, "not_satisfied", None
        ),
        "repeated-register": (
            "satisfied", 1, "not_satisfied", None
        ),
        "independent-reception-record": (
            "satisfied", 1, "satisfied", 2
        ),
        "contradictory-identity-record": (
            "satisfied", 1, "not_satisfied", None
        ),
        "late-corroboration": (
            "satisfied", 3, "satisfied", 5
        ),
        "unrelated-person": (
            "satisfied", 1, "not_satisfied", None
        ),
    }

    for case in suite["cases"]:
        name = case["expected"]["name"]
        corroboration = case["expected"]["corroboration"]

        actual = (
            name["status"],
            name["first_satisfied_turn"],
            corroboration["status"],
            corroboration["first_satisfied_turn"],
        )

        assert actual == expected[case["id"]]


def test_case_ids_unique():
    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) == len(set(ids)) == 6


def test_all_cases_use_goal05():
    assert suite["goal_id"] == "subgoal-05"
    assert set(
        item["id"]
        for item in definition["goals"]["subgoal-05"]["criteria"]
    ) == {"name", "corroboration"}


def test_evidence_does_not_exceed_completion_turn():
    for case in suite["cases"]:
        for expected in case["expected"].values():
            completion = expected["first_satisfied_turn"]

            if completion is not None:
                assert all(
                    ref["turn"] <= completion
                    for ref in expected["evidence"]
                )
