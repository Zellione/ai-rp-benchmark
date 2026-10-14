import json
import unittest
from pathlib import Path

from rpbench.judge_validation_v2 import (
    assess,
    make_payload,
    parse,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


class JudgeValidationV2Tests(unittest.TestCase):

    def setUp(self):
        self.data = json.loads(
            (ROOT / "validation/secret-leaks/v2.json")
            .read_text(encoding="utf-8")
        )
        self.config = json.loads(
            (ROOT / "configs/judges/qwen38-27b-ninfer-32k.json")
            .read_text(encoding="utf-8")
        )

    def case(self, name):
        return next(
            c for c in self.data["cases"]
            if c["id"] == name
        )

    def response(self, value):
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(value)
                },
            }],
        }

    def test_gold_dataset(self):
        validate(self.data)
        self.assertEqual(len(self.data["cases"]), 8)

    def test_two_independent_labels(self):
        expected = self.case("contamination")["expected"]
        predicted = {
            "disclosure": "direct",
            "knowledge_violation": True,
        }
        self.assertTrue(assess(expected, predicted)["exact_match"])

    def test_unauthorized_finn(self):
        case = self.case("contamination")
        value = {
            "disclosure": "direct",
            "knowledge_violation": True,
            "speaker": "Finn",
            "quote": "Mara keeps it beneath the blue lantern",
            "explanation": "Finn reveals knowledge he lacks.",
        }
        result = parse(self.response(value), case)
        self.assertTrue(result["knowledge_violation"])
        self.assertEqual(result["speaker"], "Finn")

    def test_wrong_speaker_rejected(self):
        case = self.case("contamination")
        value = {
            "disclosure": "direct",
            "knowledge_violation": True,
            "speaker": "Mara",
            "quote": "Mara keeps it beneath the blue lantern",
            "explanation": "Wrong attribution.",
        }
        with self.assertRaises(ValueError):
            parse(self.response(value), case)

    def test_authorized_transfer_context(self):
        case = self.case("authorized_transfer")
        payload = make_payload(
            case, self.data, self.config
        )
        task = json.loads(payload["messages"][1]["content"])
        self.assertEqual(len(task["prior_context"]), 1)
        self.assertEqual(
            task["turns_to_evaluate"][0]["turn"], 2
        )


if __name__ == "__main__":
    unittest.main()
