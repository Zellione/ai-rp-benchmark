import json
import tempfile
import unittest
from pathlib import Path

from rpbench.retrieval_benchmark import (
    LENGTHS,
    build_case,
    cases,
    evaluate_case,
    run_benchmark,
    stable_hash,
    summarize,
)


class RetrievalBenchmarkTests(unittest.TestCase):

    def test_case_lengths_and_gold_references(self):
        generated = list(cases())
        self.assertEqual(len(generated), 15)

        for case in generated:
            self.assertIn(case["history_length"], LENGTHS)
            self.assertEqual(
                len(case["history"]),
                case["history_length"],
            )
            self.assertEqual(
                case["target"]["turn"],
                case["history_length"] + 1,
            )

            for turn in case["gold_relevant_turns"]:
                self.assertGreaterEqual(turn, 1)
                self.assertLessEqual(
                    turn, case["history_length"]
                )

    def test_explicit_old_transfer_found(self):
        case = build_case(400, "explicit_transfer")
        result = evaluate_case(case, 12, 8)

        self.assertIn(20, result["selected_turns"])
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_no_transfer_does_not_invent_gold(self):
        case = build_case(400, "no_transfer")
        result = evaluate_case(case, 12, 8)

        self.assertTrue(result["negative_control"])
        self.assertEqual(result["gold_relevant_turns"], [])
        self.assertIsNone(result["recall_at_k"])

    def test_deterministic_generation(self):
        first = list(cases())
        second = list(cases())

        self.assertEqual(
            stable_hash(first),
            stable_hash(second),
        )

    def test_aggregation_separates_controls(self):
        results = [
            evaluate_case(case, 12, 8)
            for case in cases()
        ]
        summary = summarize(results)

        self.assertEqual(summary["total_cases"], 15)
        self.assertEqual(summary["positive_cases"], 12)
        self.assertEqual(summary["negative_controls"], 3)

    def test_run_creates_auditable_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "retrieval-validation"
            root, summary = run_benchmark(output, 12, 8)

            manifest = json.loads(
                (root / "manifest.json").read_text()
            )
            results = [
                json.loads(line)
                for line in (
                    root / "case_results.jsonl"
                ).read_text().splitlines()
            ]

            self.assertEqual(len(results), 15)
            self.assertEqual(
                manifest["retriever"],
                "lexical-retrieval-v2",
            )
            self.assertEqual(
                summary["total_cases"], 15
            )
            self.assertTrue(
                (root / "summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
