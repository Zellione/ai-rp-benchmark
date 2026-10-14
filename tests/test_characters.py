import json
import tempfile
import unittest
from pathlib import Path

from rpbench.characters import (
    build_npc_prompt,
    load_character_card,
)
from rpbench.simulation import (
    create_player_prompt,
    make_histories,
    run_simulation,
)

ROOT = Path(__file__).resolve().parents[1]


class CharacterTests(unittest.TestCase):

    def test_character_card_loads(self):
        card = load_character_card(
            ROOT / "characters/mara-vey.en.json"
        )
        self.assertEqual(card["name"], "Mara Vey")
        self.assertIn("Grey eyes", card["appearance"])

    def test_three_distinct_characters(self):
        path = ROOT / "scenarios/blackwood-selfplay.en.json"
        scenario = json.loads(path.read_text())
        prompt, cards = build_npc_prompt(scenario, path)

        self.assertEqual(len(cards), 3)
        self.assertIn("Mara Vey", prompt)
        self.assertIn("Finn Calder", prompt)
        self.assertIn("Alden Voss", prompt)
        self.assertIn("blue lantern", prompt)

    def test_player_does_not_receive_npc_secrets(self):
        scenario_path = (
            ROOT / "scenarios/blackwood-selfplay.en.json"
        )
        player = json.loads(
            (ROOT / "players/elena-ward.en.json").read_text()
        )
        scenario = json.loads(scenario_path.read_text())

        npc_prompt, _ = build_npc_prompt(
            scenario, scenario_path
        )
        player_prompt = create_player_prompt(player)

        player_messages, npc_messages = make_histories(
            player_prompt,
            npc_prompt,
            scenario["opening"],
        )

        self.assertNotIn(
            "blue lantern",
            json.dumps(player_messages).lower(),
        )
        self.assertIn(
            "blue lantern",
            json.dumps(npc_messages).lower(),
        )

    def test_manifest_archives_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_simulation(
                ROOT / "scenarios/blackwood-selfplay.en.json",
                ROOT / "players/elena-ward.en.json",
                ROOT / "configs/mock.json",
                tmp,
                turns=2,
            )
            manifest = json.loads(
                (run_dir / "manifest.json").read_text()
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["character_cards"]), 3)

            for entry in manifest["character_cards"]:
                self.assertEqual(len(entry["sha256"]), 64)

            self.assertIn(
                "blue lantern",
                manifest["rendered_npc_system_prompt"],
            )


if __name__ == "__main__":
    unittest.main()
