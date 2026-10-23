import json
import tempfile
import unittest
from pathlib import Path

from rpbench.ab_review import (
    export_html,
    prepare,
    summarize,
)


class ABReviewTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_fixture(self, name, answer, player="Hello"):
        root = self.root / name
        root.mkdir()

        (root / "manifest.json").write_text(
            json.dumps({
                "type": "goal_driven_selfplay",
                "status": "complete",
                "completed_turns": 1,
            }),
            encoding="utf-8",
        )

        entries = [
            {
                "type": "dialogue_observation",
                "turn": 1,
                "role": "player",
                "text": player,
            },
            {
                "type": "dialogue_observation",
                "turn": 1,
                "role": "npc",
                "text": answer,
            },
        ]

        with (root / "observations.jsonl").open(
            "w", encoding="utf-8"
        ) as out:
            for entry in entries:
                out.write(json.dumps(entry) + "\n")

        return root

    def test_matching_runs(self):
        a = self.run_fixture("run-a", "Good morning")
        b = self.run_fixture("run-b", "Welcome back")
        data = prepare(a, b)
        self.assertEqual(len(data["turns"]), 1)
        self.assertEqual(data["turns"][0]["player"], "Hello")
        self.assertEqual(
            {
                data["turns"][0]["A"],
                data["turns"][0]["B"],
            },
            {"Good morning", "Welcome back"},
        )
        self.assertEqual(
            data["criteria"][0], "Character fidelity"
        )

    def test_deterministic_order(self):
        a = self.run_fixture("run-a", "First")
        b = self.run_fixture("run-b", "Second")
        self.assertEqual(
            prepare(a, b)["id"],
            prepare(a, b)["id"],
        )
        self.assertEqual(
            prepare(a, b)["mapping"],
            prepare(a, b)["mapping"],
        )

    def test_mismatched_player_rejected(self):
        a = self.run_fixture("run-a", "First")
        b = self.run_fixture(
            "run-b", "Second", player="Different"
        )
        with self.assertRaisesRegex(ValueError, "Player input"):
            prepare(a, b)

    def test_identical_runs_rejected(self):
        a = self.run_fixture("run-a", "First")
        with self.assertRaises(ValueError):
            prepare(a, a)

    def test_html_export(self):
        a = self.run_fixture("run-a", "</script> text")
        b = self.run_fixture("run-b", "Second")
        output = self.root / "review.html"
        export_html(prepare(a, b), output)
        content = output.read_text(encoding="utf-8")
        self.assertIn("RPBench", content)
        self.assertNotIn("</script> text", content)
        with self.assertRaises(FileExistsError):
            export_html(prepare(a, b), output)

    def test_report(self):
        path = self.root / "results.json"
        path.write_text(json.dumps({
            "comparison_id": "test",
            "mapping": {
                "A": {"run": "model-a"},
                "B": {"run": "model-b"},
            },
            "turn_votes": [
                {"turn": 1, "vote": "A"},
                {"turn": 2, "vote": "tie"},
            ],
            "overall_preference": "A",
            "criteria": [],
        }), encoding="utf-8")
        result = summarize(path)
        self.assertEqual(result["votes"]["A"], 1)
        self.assertEqual(result["votes"]["tie"], 1)


    def fixed_fixture(self, name, context, answer):
        root = self.run_fixture(name, answer)
        (root / "cases").mkdir()

        (root / "manifest.json").write_text(
            json.dumps({
                "type": "fixed_context_npc_ab",
                "status": "complete",
                "completed_turns": 1,
            }),
            encoding="utf-8",
        )

        (root / "cases" / "001.json").write_text(
            json.dumps({
                "id": "case-1",
                "payload": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Portray Elara.",
                        },
                        {
                            "role": "user",
                            "content": context,
                        },
                        {
                            "role": "user",
                            "content": "Hello",
                        },
                    ],
                },
            }),
            encoding="utf-8",
        )
        return root

    def test_controlled_context_in_review(self):
        a = self.fixed_fixture(
            "fixed-a", "A rainy observatory", "Hello"
        )
        b = self.fixed_fixture(
            "fixed-b", "A rainy observatory", "Welcome"
        )

        data = prepare(a, b)

        self.assertEqual(
            data["turns"][0]["context"],
            [{
                "role": "user",
                "content": "A rainy observatory",
            }],
        )

    def test_different_controlled_context_rejected(self):
        a = self.fixed_fixture(
            "fixed-a", "Rain", "Hello"
        )
        b = self.fixed_fixture(
            "fixed-b", "Sunshine", "Welcome"
        )

        with self.assertRaisesRegex(
            ValueError, "identical archived inputs"
        ):
            prepare(a, b)

    def test_context_html_is_escaped(self):
        a = self.fixed_fixture(
            "fixed-a", "</script><img src=x>", "Hello"
        )
        b = self.fixed_fixture(
            "fixed-b", "</script><img src=x>", "Welcome"
        )

        output = self.root / "context.html"
        export_html(prepare(a, b), output)

        content = output.read_text(encoding="utf-8")
        self.assertNotIn("</script><img src=x>", content)
        self.assertIn("Shared scene context", content)
        self.assertIn("escapeHTML(message.content)", content)



    def test_system_prompt_variant_allowed_explicitly(self):
        a = self.fixed_fixture("variant-a", "Rain", "Response A")
        b = self.fixed_fixture("variant-b", "Rain", "Response B")

        path = b / "cases" / "001.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["payload"]["messages"][0]["content"] = (
            "Portray Elara with immersive literary prose."
        )
        path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "identical archived inputs"
        ):
            prepare(a, b)

        result = prepare(
            a, b, allow_system_prompt_difference=True
        )
        self.assertEqual(
            result["comparison_mode"], "system_prompt_variant"
        )
        self.assertNotEqual(
            result["system_prompt_comparison"]["run_a_hashes"],
            result["system_prompt_comparison"]["run_b_hashes"],
        )

    def test_system_prompt_variant_rejects_changed_history(self):
        a = self.fixed_fixture("variant-a", "Rain", "Response A")
        b = self.fixed_fixture("variant-b", "Sun", "Response B")

        with self.assertRaisesRegex(
            ValueError, "identical non-system inputs"
        ):
            prepare(a, b, allow_system_prompt_difference=True)

    def test_system_prompt_variant_rejects_changed_player(self):
        a = self.fixed_fixture("variant-a", "Rain", "Response A")
        b = self.fixed_fixture("variant-b", "Rain", "Response B")

        path = b / "cases" / "001.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["payload"]["messages"][-1]["content"] = "Goodbye"
        path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Player input differs"):
            prepare(a, b, allow_system_prompt_difference=True)

    def test_system_prompt_mode_requires_fixed_runs(self):
        a = self.run_fixture("ordinary-a", "Response A")
        b = self.run_fixture("ordinary-b", "Response B")

        with self.assertRaisesRegex(
            ValueError, "requires fixed-context runs"
        ):
            prepare(a, b, allow_system_prompt_difference=True)



if __name__ == "__main__":
    unittest.main()
