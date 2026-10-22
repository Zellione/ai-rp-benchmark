import unittest
from pathlib import Path

from rpbench.characters import build_npc_prompt
from rpbench.core import load
from rpbench.simulation import create_player_prompt
from rpbench.state import build_ground_truth, initial_state


ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "scenarios/character-chemistry"
DE = ROOT / "scenarios/character-chemistry-de"


class CharacterChemistryGermanTests(unittest.TestCase):

    def test_language_variants_have_matching_character_ids(self):
        en_scenario = load(EN / "scenario.json")
        de_scenario = load(DE / "scenario.json")

        _, en_cards = build_npc_prompt(
            en_scenario, EN / "scenario.json"
        )
        _, de_cards = build_npc_prompt(
            de_scenario, DE / "scenario.json"
        )

        self.assertEqual(
            [card["id"] for _, card in en_cards],
            [card["id"] for _, card in de_cards],
        )
        self.assertEqual(
            load(EN / "player.json")["id"],
            load(DE / "player.json")["id"],
        )

    def test_matching_character_structure(self):
        en = load(EN / "cards/elara-wynn.json")
        de = load(DE / "cards/elara-wynn.json")

        self.assertEqual(set(en), set(de))
        for key in (
            "appearance",
            "personality",
            "knowledge",
            "private_knowledge",
            "invariants",
        ):
            self.assertEqual(len(en[key]), len(de[key]))

    def test_private_knowledge_remains_separate(self):
        scenario_path = DE / "scenario.json"
        scenario = load(scenario_path)
        player = load(DE / "player.json")
        _, cards = build_npc_prompt(scenario, scenario_path)

        truth = build_ground_truth(
            player, [card for _, card in cards]
        )
        state = initial_state(truth)

        knowledge = state["character_knowledge"]["elara-wynn"]
        self.assertTrue(knowledge["private_facts"])
        self.assertTrue(knowledge["known_facts"])
        self.assertFalse(
            set(knowledge["private_facts"])
            & set(knowledge["known_facts"])
        )

    def test_german_prompt_and_player(self):
        scenario_path = DE / "scenario.json"
        scenario = load(scenario_path)

        prompt, _ = build_npc_prompt(scenario, scenario_path)
        player_prompt = create_player_prompt(
            load(DE / "player.json")
        )

        self.assertIn("idiomatisches Deutsch", prompt)
        self.assertIn("Zeitsprünge", prompt)
        self.assertIn("Rowan Vale", player_prompt)
        self.assertIn("natürliches Deutsch", player_prompt)

    def test_distinct_scenario_identity(self):
        en = load(EN / "scenario.json")
        de = load(DE / "scenario.json")

        self.assertNotEqual(en["id"], de["id"])
        self.assertTrue(de["id"].endswith("-de"))
        self.assertIn("Elara", de["opening"])


if __name__ == "__main__":
    unittest.main()
