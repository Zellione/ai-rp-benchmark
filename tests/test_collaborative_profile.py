import json
import tempfile
import unittest
from pathlib import Path

from rpbench.simulation import apply_prompt_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/qwen38-27b-ninfer-collaborative-v1.json"
)


class CollaborativeProfileTests(unittest.TestCase):

    def test_optional_legacy_behavior(self):
        result = apply_prompt_profile(
            "PLAYER BASE",
            "NPC BASE",
            {},
            CONFIG,
        )
        self.assertEqual(
            result,
            ("PLAYER BASE", "NPC BASE", None),
        )

    def test_profile_roles_and_provenance(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        player, npc, profile = apply_prompt_profile(
            "PLAYER BASE",
            "NPC BASE",
            config,
            CONFIG,
        )

        self.assertTrue(player.startswith("PLAYER BASE"))
        self.assertTrue(npc.startswith("NPC BASE"))
        self.assertIn(
            "You may initiate scene transitions",
            player,
        )
        self.assertIn(
            "The PLAYER exclusively initiates major time skips",
            npc,
        )
        self.assertIn(
            "Never write the PLAYER's dialogue",
            npc,
        )
        self.assertEqual(
            profile["id"],
            "rpbench-collaborative-v1",
        )
        self.assertEqual(len(profile["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
