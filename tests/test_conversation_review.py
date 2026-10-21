import json
import tempfile
import unittest
from pathlib import Path

from rpbench.conversation_review import (
    PAGE,
    export_html,
    prepare,
    summarize,
)


class ConversationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name, turns):
        root = self.root / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps({
                "type": "goal_driven_selfplay",
                "status": "complete",
                "completed_turns": len(turns),
            }),
            encoding="utf-8",
        )
        with (root / "observations.jsonl").open(
            "w", encoding="utf-8"
        ) as out:
            for i, (player, npc) in enumerate(turns, 1):
                for role, text in (
                    ("player", player), ("npc", npc)
                ):
                    out.write(json.dumps({
                        "type": "dialogue_observation",
                        "turn": i,
                        "role": role,
                        "text": text,
                    }) + "\n")
        return root

    def test_independent_player_dialogues(self):
        a = self.fixture("a", [
            ('"Hello."', '"Welcome."'),
            ('*I sit down.*', '"Tea?"'),
        ])
        b = self.fixture("b", [
            ('"Who are you?"', '"An archivist."'),
            ('*I walk away.*', '*She closes the door.*'),
        ])
        data = prepare(a, b, section_size=1)
        self.assertEqual(len(data["sections"]), 2)
        self.assertEqual(data["turn_counts"], {"A": 2, "B": 2})
        self.assertNotEqual(
            data["sections"][0]["A"][0]["player"],
            data["sections"][0]["B"][0]["player"],
        )

    def test_different_lengths(self):
        a = self.fixture("a", [
            ("one", "a"), ("two", "b"), ("three", "c"),
        ])
        b = self.fixture("b", [("one", "x")])
        data = prepare(a, b, section_size=2)
        self.assertEqual(len(data["sections"]), 2)
        self.assertEqual(
            sorted(data["turn_counts"].values()), [1, 3]
        )
        self.assertTrue(
            any(
                not section["A"] or not section["B"]
                for section in data["sections"]
            )
        )

    def test_order_is_stable(self):
        a = self.fixture("a", [("p", "npc a")])
        b = self.fixture("b", [("q", "npc b")])
        first = prepare(a, b)
        second = prepare(a, b)
        self.assertEqual(first, second)

    def test_reject_same_run_and_bad_section_size(self):
        a = self.fixture("a", [("p", "npc")])
        with self.assertRaises(ValueError):
            prepare(a, a)
        with self.assertRaises(ValueError):
            prepare(a, a, section_size=0)

    def test_html_escaping_and_no_overwrite(self):
        a = self.fixture("a", [
            ("*I look.*", "</script><script>alert(1)</script>")
        ])
        b = self.fixture("b", [("p", "npc")])
        output = self.root / "review.html"
        export_html(prepare(a, b), output)
        content = output.read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert(1)", content)
        self.assertIn("Conversation Review", content)
        with self.assertRaises(FileExistsError):
            export_html(prepare(a, b), output)

    def test_resume_controls(self):
        for marker in (
            'localStorage.setItem',
            'localStorage.getItem',
            'function restore(data)',
            'id="progress"',
            'id="import"',
            'id="export"',
            'id="unrated"',
        ):
            self.assertIn(marker, PAGE)

    def test_partial_report(self):
        output = self.root / "review.json"
        output.write_text(json.dumps({
            "comparison_id": "test",
            "status": "in_progress",
            "mapping": {"A": {}, "B": {}},
            "sections": [
                {"id": 1, "preference": "A"},
                {"id": 2, "preference": None},
                {"id": 3, "preference": "skip"},
            ],
            "criteria": [],
        }), encoding="utf-8")
        result = summarize(output)
        self.assertEqual(result["rated_sections"], 1)
        self.assertEqual(result["total_sections"], 3)
        self.assertEqual(result["votes"]["skip"], 1)


if __name__ == "__main__":
    unittest.main()
