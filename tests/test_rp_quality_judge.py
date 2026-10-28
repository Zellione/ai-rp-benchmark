from copy import deepcopy

import pytest

from rpbench.rp_quality_judge import (
    DIMENSIONS,
    validate_judgment,
)


TRANSCRIPT = [
    {"turn": 1, "role": "player", "text": "Ich habe eine Frage."},
    {"turn": 1, "role": "npc", "text": "Elisa legt die Pinzette ab."},
]


def valid_proposal():
    return {
        "scores": {
            key: {
                "score": 7,
                "reason": "Test assessment",
                "evidence": ["Elisa legt die Pinzette ab."],
            }
            for key in DIMENSIONS
        },
        "player_agency_violations": [],
        "major_consistency_issues": [],
        "overall_assessment": "Solid performance",
    }


def test_valid_judgment():
    assert validate_judgment(
        valid_proposal(), TRANSCRIPT
    ) == 7.0


def test_unsupported_npc_quote_rejected():
    proposal = valid_proposal()
    proposal["scores"]["german_prose"]["evidence"] = [
        "Elisa öffnet eine geheime Tür."
    ]

    with pytest.raises(ValueError, match="Unsupported"):
        validate_judgment(proposal, TRANSCRIPT)


def test_player_quote_not_accepted_as_npc_evidence():
    proposal = valid_proposal()
    proposal["scores"]["subtext"]["evidence"] = [
        "Ich habe eine Frage."
    ]

    with pytest.raises(ValueError, match="Unsupported"):
        validate_judgment(proposal, TRANSCRIPT)


def test_invalid_score_rejected():
    proposal = deepcopy(valid_proposal())
    proposal["scores"]["continuity"]["score"] = 11

    with pytest.raises(ValueError, match="Invalid score"):
        validate_judgment(proposal, TRANSCRIPT)
