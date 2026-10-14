import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.judge_validation import (
    assess,
    run_validation,
    validate_dataset,
    validation_chunk,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "validation/secret-leaks/v1.json"
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer.json"


class JudgeValidationTests(unittest.TestCase):

    def setUp(self):
        self.dataset = json.loads(
            DATASET.read_text(encoding="utf-8")
        )

    def test_dataset_is_valid(self):
        ids = validate_dataset(self.dataset)
        self.assertIn("vault-key-location", ids)
        self.assertEqual(len(self.dataset["cases"]), 8)

    def test_authorized_transfer_uses_prior_context(self):
        case = next(
            c for c in self.dataset["cases"]
            if c["id"] == "authorized_transfer"
        )
        chunk = validation_chunk(case)

        self.assertEqual(
            [t["turn"] for t in chunk["context"]], [1]
        )
        self.assertEqual(
            [t["turn"] for t in chunk["conversation"]], [2]
        )

    def test_exact_match(self):
        result = assess(
            ["indirect_leak"],
            [{"category": "indirect_leak"}],
        )
        self.assertTrue(result["exact_match"])

    def test_false_positive(self):
        result = assess(
            [],
            [{"category": "direct_leak"}],
        )
        self.assertFalse(result["exact_match"])
        self.assertEqual(
            result["false_positive_labels"],
            ["direct_leak"],
        )

    def test_missing_finding(self):
        result = assess(["direct_leak"], [])
        self.assertEqual(
            result["missed_labels"], ["direct_leak"]
        )

    def test_mock_execution_archives_every_case(self):
        def fake_chat(config, payload):
            return {
                "choices": [{
                    "message": {
                        "content": '{"findings":[]}'
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "completion_tokens": 10,
                    "prompt_tokens": 50,
                },
            }

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "judge-validation"

            with patch(
                "rpbench.judge_validation.request_chat",
                side_effect=fake_chat,
            ):
                summary = run_validation(
                    DATASET, CONFIG, out
                )

            self.assertEqual(summary["total_cases"], 8)
            self.assertEqual(summary["exact_matches"], 4)
            self.assertEqual(summary["accuracy"], 4 / 8)

            entries = [
                json.loads(line)
                for line in (
                    out / "judge_events.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                sum(
                    e["type"] == "request" for e in entries
                ),
                8,
            )
            self.assertEqual(
                sum(
                    e["type"] == "raw_response" for e in entries
                ),
                8,
            )

            self.assertEqual(
                json.loads(
                    (out / "manifest.json").read_text()
                )["status"],
                "complete",
            )


if __name__ == "__main__":
    unittest.main()
