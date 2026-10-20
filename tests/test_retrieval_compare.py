import tempfile
import unittest
from pathlib import Path

from rpbench.retrieval_bm25 import select_context
from rpbench.retrieval_benchmark_v2 import make_case
from rpbench.retrieval_compare import compare


class RetrievalCompareTests(unittest.TestCase):

    def test_bm25_finds_literal_transfer(self):
        case = make_case(400, "literal_transfer")
        result = select_context(
            case["history"], case["secret"],
            window_turns=12, evidence_turns=8,
        )
        turns = [
            entry["turn"]
            for entry in result["retrieved_older_evidence"]
        ]
        self.assertIn(80, turns)

    def test_bm25_selection_is_bounded(self):
        case = make_case(400, "negative_control")
        result = select_context(
            case["history"], case["secret"],
            window_turns=12, evidence_turns=8,
        )
        self.assertLessEqual(
            len(result["retrieved_older_evidence"]), 8
        )
        self.assertFalse(result["coverage"]["history_complete"])

    def test_comparison_preserves_same_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root, summaries = compare(
                Path(temp) / "comparison"
            )
            self.assertEqual(len(summaries), 2)
            for summary in summaries.values():
                self.assertEqual(summary["total_cases"], 18)
            self.assertTrue(
                (root / "case_results.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
