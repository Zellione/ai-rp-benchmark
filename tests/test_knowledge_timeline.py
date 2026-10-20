import copy
import unittest

from rpbench.knowledge_timeline import make_claims
from rpbench.state import initial_state, build_ground_truth


OBSERVATIONS = [
    {"turn": 1, "role": "player", "text": "Where is it?"},
    {
        "turn": 1,
        "role": "npc",
        "text": "Finn says the key is under the blue lantern.",
    },
    {"turn": 2, "role": "player", "text": "Who told you?"},
    {
        "turn": 2,
        "role": "npc",
        "text": "Mara says she remembers the hiding place.",
    },
]


def claim(turn, cid, quote):
    return {
        "kind": "knowledge_claim",
        "turn": turn,
        "character_id": cid,
        "quote": quote,
        "start_offset": None,
        "end_offset": None,
        "explanation": "Possible knowledge evidence.",
        "verified": False,
    }


class KnowledgeTimelineTests(unittest.TestCase):

    def test_claims_keep_original_turns(self):
        events = [
            claim(
                2, "mara-vey",
                "she remembers the hiding place",
            ),
            claim(
                1, "finn-calder",
                "the key is under the blue lantern",
            ),
        ]
        result = make_claims(
            OBSERVATIONS,
            events,
            {"mara-vey", "finn-calder"},
        )
        self.assertEqual(
            [e["turn"] for e in result],
            [1, 2],
        )
        self.assertEqual(
            [e["timeline_id"] for e in result],
            [1, 2],
        )

    def test_claim_does_not_confirm_transfer(self):
        result = make_claims(
            OBSERVATIONS,
            [claim(
                1, "finn-calder",
                "the key is under the blue lantern",
            )],
            {"finn-calder"},
        )
        self.assertFalse(result[0]["verified"])
        self.assertFalse(result[0]["transfer_confirmed"])
        self.assertFalse(
            result[0]["knowledge_acquisition_confirmed"]
        )
        self.assertIsNone(result[0]["source_character_id"])

    def test_fabricated_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            make_claims(
                OBSERVATIONS,
                [claim(1, "finn-calder", "invented secret")],
                {"finn-calder"},
            )

    def test_unknown_character_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            make_claims(
                OBSERVATIONS,
                [claim(
                    1, "invented",
                    "the key is under the blue lantern",
                )],
                {"finn-calder"},
            )

    def test_duplicate_claim_rejected(self):
        event = claim(
            1, "finn-calder",
            "the key is under the blue lantern",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_claims(
                OBSERVATIONS,
                [event, event],
                {"finn-calder"},
            )

    def test_nonknowledge_attribution_ignored(self):
        result = make_claims(
            OBSERVATIONS,
            [{
                "kind": "action",
                "turn": 1,
                "character_id": "finn-calder",
                "quote": "Finn says",
            }],
            {"finn-calder"},
        )
        self.assertEqual(result, [])

    def test_confirmed_state_is_not_modified(self):
        ground_truth = {
            "schema_version": 1,
            "player": {"subgoals": {}},
            "characters": {
                "finn-calder": {
                    "initial_knowledge": [],
                    "private_knowledge": [],
                    "relationships": {},
                }
            },
        }
        state = initial_state(ground_truth)
        before = copy.deepcopy(state)

        make_claims(
            OBSERVATIONS,
            [claim(
                1, "finn-calder",
                "the key is under the blue lantern",
            )],
            {"finn-calder"},
        )

        self.assertEqual(state, before)
        self.assertEqual(
            state["character_knowledge"]["finn-calder"][
                "known_facts"
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
