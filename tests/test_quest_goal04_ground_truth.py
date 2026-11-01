import json
from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.offline_goal_audit import (
    SCHEMA_VERSION,
    validate_verdict,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "tests/fixtures/quest_goal04_v4_ground_truth.json"
)
CONFIG = (
    ROOT / "experiments/simulation/quest-archivist-v1/"
    "mistral-glm-flash-player-v4-en.json"
)


def load_cases():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["cases"]


def load_definition():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return config["runtime"]["goal_criteria"]


def make_verdict(case):
    return {
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


@pytest.mark.parametrize(
    "case", load_cases(), ids=lambda case: case["id"]
)
def test_ground_truth_verdict_is_valid(case):
    definition = load_definition()
    verdict = make_verdict(case)

    validated = validate_verdict(
        verdict,
        "subgoal-04",
        definition,
        case["observations"],
    )

    for criterion in validated["criteria"]:
        expected = case["expected"][criterion["criterion_id"]]

        assert criterion["status"] == expected["status"]
        assert (
            criterion["first_satisfied_turn"]
            == expected["first_satisfied_turn"]
        )

        for evidence in criterion["evidence"]:
            matches = [
                item
                for item in case["observations"]
                if item["turn"] == evidence["turn"]
                and item["role"] == evidence["role"]
                and evidence["quote"] in item["text"]
            ]
            assert len(matches) == 1


def test_ground_truth_case_ids_unique():
    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)) == 6


def test_ground_truth_matches_v4_criterion_ids():
    definition = load_definition()
    criteria = definition["goals"]["subgoal-04"]["criteria"]

    assert [item["id"] for item in criteria] == [
        "entry", "details"
    ]

    for case in load_cases():
        assert set(case["expected"]) == {"entry", "details"}


def test_ground_truth_rejects_fabricated_quote():
    case = deepcopy(next(
        item for item in load_cases()
        if item["id"] == "complete-register-entry"
    ))

    verdict = make_verdict(case)
    verdict["criteria"][1]["evidence"][0]["quote"] = (
        "Leon Falk took the missing key at 16:20."
    )

    with pytest.raises(ValueError, match="source mismatch"):
        validate_verdict(
            verdict,
            "subgoal-04",
            load_definition(),
            case["observations"],
        )


def test_ground_truth_rejects_premature_completion():
    case = deepcopy(next(
        item for item in load_cases()
        if item["id"] == "complete-register-entry"
    ))

    verdict = make_verdict(case)
    verdict["criteria"][1]["first_satisfied_turn"] = 1

    with pytest.raises(ValueError, match="precedes"):
        validate_verdict(
            verdict,
            "subgoal-04",
            load_definition(),
            case["observations"],
        )


def test_ground_truth_requires_valid_status():
    for case in load_cases():
        for expected in case["expected"].values():
            assert expected["status"] in {
                "satisfied", "partial", "not_satisfied"
            }
            if expected["status"] == "satisfied":
                assert type(expected["first_satisfied_turn"]) is int
            else:
                assert expected["first_satisfied_turn"] is None
