import tempfile
import unittest

from pathlib import Path

from rpbench.reviewer_tui import (
    Reviewer,
    build_display,
    load_turns,
    wrap,
)
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


class ReviewerTUITests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.run_dir = run_simulation(
            ROOT / "scenarios/blackwood-selfplay.en.json",
            ROOT / "players/elena-ward.en.json",
            ROOT / "configs/mock.json",
            self.temp.name,
            turns=3,
        )

    def test_load_turns(self):
        turns = load_turns(self.run_dir)

        self.assertEqual(len(turns), 3)
        self.assertIn("player", turns[1])
        self.assertIn("npc", turns[1])

    def test_display_contains_both_roles(self):
        turns = load_turns(self.run_dir)
        text = build_display(1, turns[1])

        self.assertIn("PLAYER", text)
        self.assertIn("NPC", text)
        self.assertIn("TURN 1", text)

    def test_navigation_bounds(self):
        reviewer = Reviewer(
            self.run_dir, "test-reviewer"
        )

        reviewer.move(100)
        self.assertEqual(reviewer.current_turn(), 3)

        reviewer.move(-100)
        self.assertEqual(reviewer.current_turn(), 1)

    def test_wrapping_preserves_text(self):
        lines = wrap("one two three four", 8)

        self.assertGreater(len(lines), 1)
        self.assertEqual(
            " ".join(lines), "one two three four"
        )


if __name__ == "__main__":
    unittest.main()
