from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.core import Backend
from rpbench.offline_goal_audit import SCHEMA_VERSION
from rpbench.judge_calibration import (
    calibrate_case,
    compare,
    load_suite,
    make_identity,
    run_suite,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = ROOT / (
    "tests/fixtures/quest_goal04_v4_ground_truth.json"
)

DEFINITION = ROOT / (
    "experiments/simulation/quest-archivist-v1/"
    "mistral-glm-flash-player-v4-en.json"
)

MODEL = "fixture-model"
BASE_URL = "https://nano-gpt.com/api/subscription"


def setup_case():
    suite, definition = load_suite(FIXTURE, DEFINITION)
    case = next(
        item for item in suite["cases"]
        if item["id"] == "complete-register-entry"
    )
    return suite, definition, case


def test_all_reference_cases_load():
    suite, definition = load_suite(FIXTURE, DEFINITION)
    assert len(suite["cases"]) == 6
    assert len(definition["goals"]["subgoal-04"]["criteria"]) == 2


def test_dry_run_does_not_create_output(tmp_path):
    target = tmp_path / "calibration"

    result = run_suite(
        FIXTURE, DEFINITION, target, MODEL
    )

    assert len(result) == 6
    assert all(item["status"] == "dry_run" for item in result)
    assert not target.exists()


def test_identity_changes_when_criteria_change():
    suite, definition, case = setup_case()

    original = make_identity(
        suite, definition, case, MODEL, BASE_URL, 16384
    )

    changed = deepcopy(definition)
    changed["goals"]["subgoal-04"]["criteria"][0][
        "description"
    ] += " Additional requirement."

    revised = make_identity(
        suite, changed, case, MODEL, BASE_URL, 16384
    )

    assert original["definition_sha256"] != (
        revised["definition_sha256"]
    )


def test_comparison_detects_wrong_completion_turn():
    _, _, case = setup_case()

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

    verdict["criteria"][0]["first_satisfied_turn"] = 1

    result = compare(case, verdict)

    assert result[0]["status_match"]
    assert not result[0]["turn_match"]
    assert result[1]["turn_match"]


def test_cached_raw_response_does_not_repeat_api_call(
    tmp_path, monkeypatch
):
    import json

    suite, definition, case = setup_case()

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

    calls = []

    def fake_generate(self, url, payload):
        calls.append(url)
        return json.dumps(verdict), {
            "choices": [{"finish_reason": "stop"}],
            "usage": {"total_tokens": 100},
        }

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")
    monkeypatch.setattr(Backend, "generate", fake_generate)

    output = tmp_path / "calibration"

    kwargs = dict(
        suite=suite,
        definition=definition,
        case=case,
        output=output,
        model=MODEL,
        base_url=BASE_URL,
        max_tokens=16384,
        execute=True,
    )

    first = calibrate_case(**kwargs)

    assert first["status"] == "valid"
    assert all(
        row["status_match"] and row["turn_match"]
        for row in first["comparison"]
    )

    monkeypatch.delenv("NANOGPT_API_KEY")
    (output / "result.json").unlink()

    second = calibrate_case(**kwargs)

    assert second["status"] == "valid"
    assert len(calls) == 1

    summary = summarize([second])
    assert summary["status_accuracy"] == 1.0
    assert summary["satisfied_turn_correct"] == 2

    with pytest.raises(ValueError, match="identity mismatch"):
        calibrate_case(**{**kwargs, "model": "changed-model"})



def test_invalid_cached_result_is_revalidated(
    tmp_path, monkeypatch
):
    import json

    suite, definition, case = setup_case()

    calls = []

    def fake_generate(self, url, payload):
        calls.append(url)
        return json.dumps({
            "schema_version": SCHEMA_VERSION,
            "goal_id": "subgoal-04",
            "criteria": [],
        }), {
            "choices": [{"finish_reason": "stop"}],
        }

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")
    monkeypatch.setattr(Backend, "generate", fake_generate)

    kwargs = {
        "suite": suite,
        "definition": definition,
        "case": case,
        "output": tmp_path / "calibration",
        "model": MODEL,
        "base_url": BASE_URL,
        "max_tokens": 16384,
        "execute": True,
    }

    first = calibrate_case(**kwargs)
    assert first["status"] == "invalid"

    monkeypatch.delenv("NANOGPT_API_KEY")

    second = calibrate_case(**kwargs)

    assert second["status"] == "invalid"
    assert len(calls) == 1


def test_summary_counts_invalid_case_criteria():
    results = [
        {
            "case_id": "valid-case",
            "status": "valid",
            "comparison": [
                {
                    "criterion_id": "entry",
                    "expected_status": "satisfied",
                    "actual_status": "satisfied",
                    "status_match": True,
                    "expected_turn": 1,
                    "actual_turn": 1,
                    "turn_match": True,
                },
            ],
            "normalizations": [],
            "error": None,
        },
        {
            "case_id": "invalid-case",
            "status": "invalid",
            "expected_criterion_ids": ["entry", "details"],
            "comparison": [],
            "normalizations": [],
            "error": "Invalid JSON",
        },
    ]

    summary = summarize(results)

    assert summary["cases"] == 2
    assert summary["valid_cases"] == 1
    assert summary["invalid_cases"] == 1
    assert summary["criteria_scored"] == 1
    assert summary["criteria_total"] == 3



def test_end_to_end_accuracy_penalizes_invalid_cases():
    results = [
        {
            "case_id": "correct",
            "status": "valid",
            "comparison": [
                {
                    "criterion_id": "entry",
                    "expected_status": "satisfied",
                    "actual_status": "satisfied",
                    "status_match": True,
                    "expected_turn": 2,
                    "actual_turn": 2,
                    "turn_match": True,
                },
            ],
            "normalizations": [],
            "error": None,
        },
        {
            "case_id": "invalid",
            "status": "invalid",
            "expected_criterion_ids": ["entry", "details"],
            "comparison": [],
            "normalizations": [],
            "error": "Invalid evidence",
        },
    ]

    summary = summarize(results)

    assert summary["status_accuracy"] == 1.0
    assert summary["criteria_total"] == 3
    assert summary["end_to_end_correct"] == 1
    assert summary["end_to_end_accuracy"] == pytest.approx(1 / 3)
    assert summary["perfect_cases"] == 1
    assert summary["perfect_case_rate"] == 0.5


def test_end_to_end_accuracy_penalizes_wrong_turn():
    results = [{
        "case_id": "wrong-turn",
        "status": "valid",
        "comparison": [{
            "criterion_id": "entry",
            "expected_status": "satisfied",
            "actual_status": "satisfied",
            "status_match": True,
            "expected_turn": 2,
            "actual_turn": 3,
            "turn_match": False,
        }],
        "normalizations": [],
        "error": None,
    }]

    summary = summarize(results)

    assert summary["status_accuracy"] == 1.0
    assert summary["end_to_end_accuracy"] == 0.0
    assert summary["perfect_cases"] == 0
