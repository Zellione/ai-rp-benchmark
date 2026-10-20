import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.attribution_validation import (
    validate_dataset,
    normalize_event,
    normalized_events,
    score,
    metrics,
    parse_response,
    run_validation,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "validation/attribution/v1.json"
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"


def response(events):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": json.dumps({"events": events})
            },
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
    }


class AttributionValidationTests(unittest.TestCase):

    def setUp(self):
        self.data = json.loads(
            DATASET.read_text(encoding="utf-8")
        )
        self.ids = validate_dataset(self.data)

    def case(self, name):
        return next(
            c for c in self.data["cases"]
            if c["id"] == name
        )

    def test_gold_dataset(self):
        self.assertEqual(len(self.data["cases"]), 18)
        self.assertEqual(len(self.ids), 3)

    def test_multi_speaker_gold(self):
        case = self.case("two-speakers")
        events = normalized_events(
            case["events"], case["text"], self.ids
        )
        self.assertEqual(
            [e["speaker_id"] for e in events],
            ["mara-vey", "finn-calder"],
        )

    def test_repeated_quote_occurrences(self):
        case = self.case("repeated-phrase")
        events = normalized_events(
            case["events"], case["text"], self.ids
        )
        self.assertNotEqual(
            events[0]["start_offset"],
            events[1]["start_offset"],
        )

    def test_invalid_character_rejected(self):
        case = self.case("named-action")
        event = {
            **case["events"][0],
            "actor_id": "invented",
        }
        with self.assertRaises(ValueError):
            normalize_event(event, case["text"], self.ids)

    def test_narrator_cannot_be_speaker(self):
        case = self.case("third-person-narration")
        event = {
            **case["events"][0],
            "speaker_id": "mara-vey",
        }
        with self.assertRaises(ValueError):
            normalize_event(event, case["text"], self.ids)

    def test_exact_match_and_wrong_subject(self):
        case = self.case("subject-not-speaker")
        gold = normalized_events(
            case["events"], case["text"], self.ids
        )
        self.assertTrue(score(gold, gold)["exact_match"])

        wrong = [
            {**gold[0], "subject_id": "mara-vey"}
        ]
        result = score(gold, wrong)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)

    def test_missing_event_reduces_recall(self):
        case = self.case("explicit-mara-speech")
        gold = normalized_events(
            case["events"], case["text"], self.ids
        )
        result = score(gold, [])
        self.assertEqual(result["fn"], 1)
        self.assertFalse(result["exact_match"])

    def test_response_parser(self):
        case = self.case("explicit-finn-speech")
        result = parse_response(
            response(case["events"]), case, self.ids
        )
        self.assertEqual(result[0]["speaker_id"], "finn-calder")

    def test_mock_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "validation"

            with patch(
                "rpbench.attribution_validation.request_chat",
                side_effect=lambda cfg, payload: response([]),
            ) as mock:
                summary = run_validation(
                    DATASET, CONFIG, out
                )

            self.assertEqual(mock.call_count, 18)
            self.assertEqual(summary["cases"], 18)
            self.assertEqual(summary["tp"], 0)
            self.assertEqual(
                summary["exact_case_matches"], 1
            )
            self.assertTrue(
                (out / "case_results.jsonl").is_file()
            )
            manifest = json.loads(
                (out / "manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
