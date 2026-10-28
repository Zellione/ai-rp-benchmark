from collections import Counter

from tools.build_refusal_pilot import (
    PROBES,
    build_scenario,
)


def test_refusal_pilot_has_sixteen_unique_cases():
    assert len(PROBES) == 16
    assert len({entry[0] for entry in PROBES}) == 16


def test_refusal_pilot_category_distribution():
    counts = Counter(entry[1] for entry in PROBES)
    assert counts == {
        "benign_trigger": 4,
        "dark_fiction": 4,
        "sexual_content": 4,
        "safety_boundary": 4,
    }


def test_sexual_content_has_both_expected_behaviors():
    expectations = [
        entry[2]
        for entry in PROBES
        if entry[1] == "sexual_content"
    ]
    assert expectations.count("continue") == 2
    assert expectations.count("boundary") == 2


def test_all_probes_use_fixed_three_turn_history():
    for entry in PROBES:
        scenario = build_scenario(entry)
        assert scenario["probe_turn"] == 3
        assert len(scenario["turns"]) == 3
        assert scenario["turns"][2] == entry[4]
        assert scenario["expected_behavior"] in {
            "continue", "boundary"
        }
        assert scenario["system"].strip()


def test_no_probe_has_empty_messages():
    for entry in PROBES:
        scenario = build_scenario(entry)
        assert all(text.strip() for text in scenario["turns"])
