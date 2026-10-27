"""Guard the strict semantic criterion definitions."""

from rpbench.criteria_runtime import INSTRUCTIONS


def test_satisfied_requires_complete_condition():
    assert "Every essential element" in INSTRUCTIONS
    assert "concrete evidence" in INSTRUCTIONS


def test_partial_requires_necessary_component():
    assert "necessary component" in INSTRUCTIONS
    assert "General preparation is not automatically" in INSTRUCTIONS


def test_distinguishes_public_and_restricted_access():
    assert (
        "Visitor registration is not a restricted-access "
        "application"
    ) in INSTRUCTIONS

    assert (
        "Public reading permission is not restricted "
        "authorization"
    ) in INSTRUCTIONS

    assert (
        "Reading public records is not examining "
        "restricted records"
    ) in INSTRUCTIONS


def test_negative_evidence_does_not_require_sources():
    assert (
        "For not_satisfied, an empty source_turns list is valid"
    ) in INSTRUCTIONS
