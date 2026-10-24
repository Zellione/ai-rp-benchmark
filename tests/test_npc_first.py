import tempfile
import unittest
from pathlib import Path

from rpbench.simulation import make_histories
from rpbench.resume import _histories, _sync_transcript


OPENING = '*Elara öffnet die Bürotür.* "Du bist zurück."'


class NpcFirstTests(unittest.TestCase):
    def test_npc_first_histories(self):
        player, npc = make_histories(
            "Player system",
            "NPC system",
            "Legacy opening",
            OPENING,
        )
        self.assertEqual(
            [m["role"] for m in player],
            ["system", "user"],
        )
        self.assertEqual(player[1]["content"], OPENING)
        self.assertEqual(
            [m["role"] for m in npc],
            ["system", "assistant"],
        )
        self.assertEqual(npc[1]["content"], OPENING)

    def test_legacy_histories_unchanged(self):
        player, npc = make_histories(
            "Player system", "NPC system", "Legacy opening"
        )
        self.assertEqual(player[1]["content"], "Legacy opening")
        self.assertEqual(len(npc), 1)

    def test_reject_empty_npc_opening(self):
        with self.assertRaisesRegex(ValueError, "npc_opening"):
            make_histories("player", "npc", "legacy", "")

    def test_resume_reconstructs_opening(self):
        manifest = {
            "scenario": {
                "opening": "Legacy opening",
                "npc_opening": OPENING,
            },
            "rendered_player_system_prompt": "Player",
            "rendered_npc_system_prompt": "NPC",
        }
        entries = [
            {"turn": 1, "role": "player", "text": '"Hallo."'},
            {"turn": 1, "role": "npc", "text": '"Komm rein."'},
        ]
        player, npc = _histories(manifest, entries)
        self.assertEqual(player[1]["content"], OPENING)
        self.assertEqual(npc[1]["content"], OPENING)
        self.assertEqual(npc[-1]["content"], '"Komm rein."')

    def test_transcript_recovery_includes_opening(self):
        manifest = {
            "scenario": {
                "id": "test",
                "title": "Test",
                "npc_opening": OPENING,
            },
            "player": {"name": "Rowan"},
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "conversation.md"
            path.write_text(
                "# Test\n\nPlayer: Rowan\n\n"
                "Mode: Goal-driven self-play\n\n",
                encoding="utf-8",
            )
            count = _sync_transcript(root, manifest, [])
            self.assertEqual(count, 0)
            self.assertIn(
                f"## NPC Opening\n\n{OPENING}",
                path.read_text(encoding="utf-8"),
            )
            before = path.read_bytes()
            _sync_transcript(root, manifest, [])
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
