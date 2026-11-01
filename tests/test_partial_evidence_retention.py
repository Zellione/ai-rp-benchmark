from rpbench.partial_evidence import prior_partial_evidence
from rpbench.criteria_runtime import INSTRUCTIONS


def make_progress(turns):
    return {
        "evidence": [
            {
                "turn": turn,
                "status": "partial",
                "reason": f"Evidence from turn {turn}",
            }
            for turn in turns
        ]
    }


def make_observations(turns):
    return [
        {
            "turn": turn,
            "role": "npc",
            "text": f"NPC observation {turn}",
        }
        for turn in turns
    ]


def test_earliest_partial_survives_many_updates():
    turns = [6, 7, 8, 9, 10, 14, 15]

    result = prior_partial_evidence(
        make_progress(turns),
        make_observations(turns),
        current_turn=16,
    )

    assert [item["turn"] for item in result] == [6, 14, 15]


def test_short_partial_history_is_preserved():
    turns = [2, 4]

    result = prior_partial_evidence(
        make_progress(turns),
        make_observations(turns),
        current_turn=5,
    )

    assert [item["turn"] for item in result] == [2, 4]


def test_single_slot_retains_earliest_evidence():
    turns = [2, 4, 6]

    result = prior_partial_evidence(
        make_progress(turns),
        make_observations(turns),
        current_turn=7,
        limit=1,
    )

    assert [item["turn"] for item in result] == [2]


def test_future_evidence_is_excluded():
    turns = [2, 4, 6]

    result = prior_partial_evidence(
        make_progress(turns),
        make_observations(turns),
        current_turn=6,
    )

    assert [item["turn"] for item in result] == [2, 4]


def test_downgrade_requires_reassessment_of_prior_evidence():
    assert "absence of NEW evidence alone" in INSTRUCTIONS
    assert "contradicted or retracted" in INSTRUCTIONS
