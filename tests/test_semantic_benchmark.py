"""Semantic benchmark fixtures and evaluation boundaries."""

from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.core import load
from rpbench.semantic_benchmark import (
    case_inputs, run_case, validate_suite,
)


SUITE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/semantic/blackwood-v1.json"
)


def test_suite_and_case_count():
    suite = validate_suite(load(SUITE))
    assert len(suite["cases"]) == 7
    assert len({case["id"] for case in suite["cases"]}) == 7


def test_partial_evidence_is_injected():
    case = next(
        c for c in load(SUITE)["cases"]
        if c["id"] == "voss-cumulative"
    )

    _, _, progress = case_inputs(case)
    evidence = progress["goals"]["subgoal-01"][
        "criteria"
    ]["voss-role"]["evidence"]

    assert evidence == [{
        "turn": 1,
        "status": "partial",
        "reason": case["prior_partials"][0]["reason"],
    }]


def test_duplicate_case_rejected():
    suite = deepcopy(load(SUITE))
    suite["cases"].append(deepcopy(suite["cases"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_suite(suite)


def test_correct_cumulative_result(monkeypatch):
    from rpbench import semantic_benchmark

    case = load(SUITE)["cases"][0]

    def fake_generate(
        definition, progress, observations, settings,
        **kwargs,
    ):
        kwargs["on_response"]({
            "finish_reason": "stop",
            "usage": {"completion_tokens": 15},
            "content": '{"schema_version":1}',
            "reasoning": "",
        })
        return {
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-01",
                "criterion_id": "voss-role",
                "status": "satisfied",
                "source_turns": [1, 2],
                "reason": "Combined curator responsibility and identity.",
            }],
        }, {"finish_reason": "stop"}

    monkeypatch.setattr(
        semantic_benchmark, "_generate", fake_generate
    )

    result = run_case(case, {})
    assert result["valid"]
    assert result["correct"]
    assert result["source_turns"] == [1, 2]


def test_false_source_reference_rejected(monkeypatch):
    from rpbench import semantic_benchmark

    case = load(SUITE)["cases"][0]

    def fake_generate(
        definition, progress, observations, settings,
        **kwargs,
    ):
        return {
            "schema_version": 1,
            "criterion_updates": [{
                "goal_id": "subgoal-01",
                "criterion_id": "voss-role",
                "status": "satisfied",
                "source_turns": [1, 99],
                "reason": "Fabricated turn.",
            }],
        }, {"finish_reason": "stop"}

    monkeypatch.setattr(
        semantic_benchmark, "_generate", fake_generate
    )

    result = run_case(case, {})
    assert not result["valid"]
    assert not result["correct"]
