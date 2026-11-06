import pytest

from rpbench.false_positive_audit import (
    classify,
    validate_verdict,
)


DEFINITION = {
    "goals": {
        "subgoal-01": {
            "criteria": [
                {"id": "a"},
                {"id": "b"},
            ],
        },
    },
}

PREFIX = [
    {"turn": 1, "role": "player", "text": "Request"},
    {"turn": 1, "role": "npc", "text": "Response"},
]


def verdict(a="satisfied", b="satisfied"):
    return {
        "schema_version": 1,
        "goal_id": "subgoal-01",
        "criteria": [
            {
                "criterion_id": cid,
                "status": status,
                "evidence": [{
                    "turn": 1,
                    "role": "npc",
                    "summary": "Semantic description, not a quote",
                }],
                "reason": "Explanation",
            }
            for cid, status in (("a", a), ("b", b))
        ],
    }


def test_all_satisfied_is_valid():
    result = validate_verdict(
        verdict(), "subgoal-01", DEFINITION, PREFIX
    )
    assert classify(result) == "valid"


def test_partial_online_completion_is_false_positive():
    result = validate_verdict(
        verdict(b="partial"),
        "subgoal-01", DEFINITION, PREFIX,
    )
    assert classify(result) == "false_positive"


def test_uncertain_requires_review():
    assert classify(verdict(b="uncertain")) == "uncertain"


def test_future_evidence_is_rejected():
    proposal = verdict()
    proposal["criteria"][0]["evidence"][0]["turn"] = 2

    with pytest.raises(
        ValueError, match="provenance"
    ):
        validate_verdict(
            proposal, "subgoal-01", DEFINITION, PREFIX
        )
