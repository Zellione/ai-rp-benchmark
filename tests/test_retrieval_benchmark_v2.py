import tempfile
import unittest
from pathlib import Path

from rpbench.retrieval_benchmark_v2 import (
    cases,
    evaluate_case,
    make_case,
    run_benchmark,
)
from rpbench.retrieval_benchmark import stable_hash


class RetrievalBenchmarkV2Tests(unittest.TestCase):

    def test_dataset_has_18_cases(self):
        self.assertEqual(len(list(cases())), 18)

    def test_deterministic(self):
        self.assertEqual(
            stable_hash(list(cases())),
            stable_hash(list(cases())),
        )

    def test_literal_transfer(self):
        result = evaluate_case(
            make_case(400, "literal_transfer")
        )
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_semantic_opaque_reference(self):
        case = make_case(400, "opaque_reference")
        result = evaluate_case(case)

        self.assertEqual(len(case["gold_relevant_turns"]), 1)
        self.assertIsNotNone(result["recall_at_k"])

    def test_multi_hop_has_two_gold_turns(self):
        case = make_case(400, "multi_hop")
        self.assertEqual(len(case["gold_relevant_turns"]), 2)

    def test_no_transfer_negative(self):
        result = evaluate_case(
            make_case(400, "negative_control")
        )
        self.assertTrue(result["negative_control"])
        self.assertIsNone(result["recall_at_k"])

    def test_run_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root, summary = run_benchmark(
                Path(temp) / "benchmark"
            )
            self.assertEqual(summary["total_cases"], 18)
            self.assertTrue(
                (root / "case_results.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
