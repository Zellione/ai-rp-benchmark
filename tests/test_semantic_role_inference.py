"""Role inference semantics and benchmark regression cases."""

import json
from pathlib import Path

from rpbench.criteria_runtime import INSTRUCTIONS
from rpbench.semantic_benchmark import validate_suite


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_accepts_supported_inference():
    assert "player-accessible" in INSTRUCTIONS
    assert "Do not require an exact title" in INSTRUCTIONS


def test_prompt_rejects_weak_role_inference():
    assert "plausible alternatives" in INSTRUCTIONS
    assert "isolated instructions" in INSTRUCTIONS


def test_existing_authorization_safeguards_preserved():
    assert (
        "Submitting an application is not receiving "
        "authorization."
    ) in INSTRUCTIONS
    assert (
        "Hypothetical future actions are not completed "
        "actions."
    ) in INSTRUCTIONS


def test_blackwood_v2_preserves_v1():
    v1 = json.loads(
        (ROOT / "benchmarks/semantic/blackwood-v1.json")
        .read_text(encoding="utf-8")
    )
    v2 = json.loads(
        (ROOT / "benchmarks/semantic/blackwood-v2.json")
        .read_text(encoding="utf-8")
    )

    validate_suite(v1)
    validate_suite(v2)

    assert v2["cases"][:len(v1["cases"])] == v1["cases"]
    assert len(v2["cases"]) == len(v1["cases"]) + 4


def test_mara_regression_gold_labels():
    suite = json.loads(
        (ROOT / "benchmarks/semantic/blackwood-v2.json")
        .read_text(encoding="utf-8")
    )

    labels = {
        case["id"]: case["expected_status"]
        for case in suite["cases"]
    }

    assert labels["mara-strong-implicit"] == "satisfied"
    assert labels["mara-weak-implicit"] == "partial"
    assert labels["mara-explicit-title"] == "satisfied"
    assert labels["mara-contradicted-role"] == "partial"
