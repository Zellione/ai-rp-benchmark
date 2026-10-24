import unittest

from rpbench.longform_evidence import validate_finding


OBSERVATIONS = [
    {
        "turn": 1,
        "role": "player",
        "text": "*Rowan takes off his wet shirt.*",
    },
    {
        "turn": 1,
        "role": "npc",
        "text": '"Here is a towel."',
    },
    {
        "turn": 2,
        "role": "player",
        "text": "*Rowan puts the towel over his shoulders.*",
    },
    {
        "turn": 2,
        "role": "npc",
        "text": "*Elara straightens the shirt he is wearing.*",
    },
]


class LongformEvidenceTests(unittest.TestCase):
    def finding(self, category="world_state", verdict="candidate"):
        return {
            "category": category,
            "verdict": verdict,
            "reason": "Possible clothing-state contradiction",
            "evidence": [
                {
                    "turn": 1,
                    "role": "player",
                    "quote": "takes off his wet shirt",
                },
                {
                    "turn": 2,
                    "role": "npc",
                    "quote": "the shirt he is wearing",
                },
            ],
        }

    def test_valid_evidence(self):
        validate_finding(self.finding(), OBSERVATIONS)

    def test_missing_quote_rejected(self):
        finding = self.finding()
        finding["evidence"][1]["quote"] = "puts his shirt back on"

        with self.assertRaisesRegex(ValueError, "not found"):
            validate_finding(finding, OBSERVATIONS)

    def test_unknown_category_rejected(self):
        with self.assertRaisesRegex(ValueError, "category"):
            validate_finding(
                self.finding(category="unusual_behavior"),
                OBSERVATIONS,
            )

    def test_one_turn_character_drift_not_confirmed(self):
        finding = self.finding(
            category="character_development",
            verdict="confirmed",
        )

        with self.assertRaisesRegex(
            ValueError, "at least three distinct turns"
        ):
            validate_finding(finding, OBSERVATIONS)

    def test_unusual_character_behavior_can_be_candidate(self):
        validate_finding(
            self.finding(category="character_development"),
            OBSERVATIONS,
        )

    def test_chronological_evidence_required(self):
        finding = self.finding()
        finding["evidence"].reverse()

        with self.assertRaisesRegex(ValueError, "chronological"):
            validate_finding(finding, OBSERVATIONS)


if __name__ == "__main__":
    unittest.main()
