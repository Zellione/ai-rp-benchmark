import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.knowledge_transfer_validation import (
    metrics,
    run_validation,
    score,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "validation/knowledge-transfers/v1.json"
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"


def response(transfers):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": json.dumps({
                    "transfers": transfers
                })
            },
        }],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 10,
        },
    }


class TransferValidationTests(unittest.TestCase):

    def setUp(self):
        self.data = json.loads(
            DATASET.read_text(encoding="utf-8")
        )

    def test_dataset(self):
        ids = validate_dataset(self.data)
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(self.data["cases"]), 12)

    def test_gold_has_positive_and_negative_cases(self):
        positives = sum(
            bool(case["expected"])
            for case in self.data["cases"]
        )
        self.assertEqual(positives, 6)

    def test_exact_match(self):
        event = self.data["cases"][0]["expected"][0]
        self.assertTrue(
            score([event], [event])["exact_match"]
        )

    def test_wrong_recipient_is_error(self):
        event = self.data["cases"][0]["expected"][0]
        incorrect = {
            **event,
            "recipient_character_id": "alden-voss",
        }
        result = score([event], [incorrect])
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)

    def test_unknown_prior_knowledge_is_distinct(self):
        event = self.data["cases"][0]["expected"][0]
        incorrect = {
            **event,
            "recipient_prior_knowledge": "known",
        }
        self.assertFalse(
            score([event], [incorrect])["exact_match"]
        )

    def test_metrics(self):
        result = metrics([
            {"tp": 1, "fp": 0, "fn": 0, "exact_match": True},
            {"tp": 0, "fp": 1, "fn": 1, "exact_match": False},
        ])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_mock_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation"
            with patch(
                "rpbench.knowledge_transfer_validation.request_chat",
                return_value=response([]),
            ) as mock:
                summary = run_validation(
                    DATASET, CONFIG, path
                )

            self.assertEqual(mock.call_count, 12)
            self.assertEqual(summary["cases"], 12)
            self.assertEqual(summary["exact_case_matches"], 6)
            self.assertEqual(summary["tp"], 0)
            self.assertEqual(
                json.loads(
                    (path / "manifest.json").read_text()
                )["status"],
                "complete",
            )


if __name__ == "__main__":
    unittest.main()
