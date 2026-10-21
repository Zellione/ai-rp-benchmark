import json
import tempfile
import unittest
from pathlib import Path

from rpbench.ab_review import PAGE, summarize


class ABReviewResumeTests(unittest.TestCase):

    def test_resume_controls_present(self):
        for marker in (
            'id="review-progress"',
            'id="import-file"',
            'id="review-status"',
            'id="jump"',
            'id="open-turn"',
            'localStorage.setItem',
            'localStorage.getItem',
            'function restore(data)',
            'comparison_id: DATA.id',
            'mapping: DATA.mapping',
        ):
            self.assertIn(marker, PAGE)

    def test_partial_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "comparison_id": "personal-test",
                "status": "in_progress",
                "mapping": {
                    "A": {"run": "a"},
                    "B": {"run": "b"},
                },
                "turn_votes": [
                    {"turn": 1, "vote": "A"},
                    {"turn": 2, "vote": None},
                    {"turn": 3, "vote": "skip"},
                ],
                "criteria": [],
                "overall_preference": None,
                "final_notes": "",
            }), encoding="utf-8")
            result = summarize(path)
            self.assertEqual(result["status"], "in_progress")
            self.assertEqual(result["rated_turns"], 1)
            self.assertEqual(result["total_turns"], 3)
            self.assertAlmostEqual(
                result["completion_fraction"], 1 / 3
            )

    def test_old_export_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.json"
            path.write_text(json.dumps({
                "comparison_id": "old",
                "mapping": {"A": {}, "B": {}},
                "turn_votes": [
                    {"turn": 1, "vote": "B"},
                ],
                "criteria": [],
            }), encoding="utf-8")
            result = summarize(path)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["votes"]["B"], 1)


if __name__ == "__main__":
    unittest.main()
