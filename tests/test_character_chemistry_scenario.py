import unittest
from pathlib import Path

from rpbench.characters import build_npc_prompt
from rpbench.core import load
from rpbench.simulation import create_player_prompt
from rpbench.state import build_ground_truth, initial_state


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = (
    ROOT / "scenarios/character-chemistry/scenario.json"
)
PLAYER = (
    ROOT / "scenarios/character-chemistry/player.json"
)


class CharacterChemistryScenarioTests(unittest.TestCase):

    def test_cards_load(self):
        scenario = load(SCENARIO)
        prompt, cards = build_npc_prompt(
            scenario, SCENARIO
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0][1]["id"], "elara-wynn")
        self.assertIn("Elara Wynn", prompt)

    def test_private_knowledge_is_separate(self):
        scenario = load(SCENARIO)
        _, cards = build_npc_prompt(scenario, SCENARIO)
        truth = build_ground_truth(
            load(PLAYER),
            [card for _, card in cards],
        )
        state = initial_state(truth)

        knowledge = state[
            "character_knowledge"
        ]["elara-wynn"]

        self.assertTrue(knowledge["private_facts"])
        self.assertTrue(knowledge["known_facts"])
        self.assertNotIn(
            knowledge["private_facts"][0],
            knowledge["known_facts"],
        )

    def test_player_prompt(self):
        prompt = create_player_prompt(load(PLAYER))

        self.assertIn("Rowan Vale", prompt)
        self.assertIn("quotation marks", prompt)
        self.assertIn("asterisks", prompt)

    def test_roleplay_format_instructions(self):
        system = load(SCENARIO)["system"]

        self.assertIn("FORMATTING:", system)
        self.assertIn("AGENCY:", system)
        self.assertIn("PACING:", system)
        self.assertIn("CONSISTENCY:", system)

    def test_opening_has_no_forced_resolution(self):
        opening = load(SCENARIO)["opening"]

        self.assertIn("observatory", opening)
        self.assertIn("Rowan", opening)
        self.assertIn("Elara", opening)


if __name__ == "__main__":
    unittest.main()
