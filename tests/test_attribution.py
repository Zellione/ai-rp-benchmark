import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.attribution import (
    character_catalog,
    evaluate,
    parse_response,
)
from rpbench.core import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"

CATALOG = [
    {"id": "mara-vey", "name": "Mara Vey"},
    {"id": "finn", "name": "Finn"},
]

JOB = {
    "turn": 3,
    "text": (
        'Mara glances at the lantern. '
        '"I know where it is," Finn says.'
    ),
}


def reply(events):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": json.dumps({"events": events})
            },
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40,
        },
    }


class AttributionTests(unittest.TestCase):

    def test_speech_and_action(self):
        events = parse_response(reply([
            {
                "kind": "action",
                "character_id": "mara-vey",
                "quote": "Mara glances at the lantern.",
                "explanation": "Explicitly attributed gesture.",
            },
            {
                "kind": "speech",
                "character_id": "finn",
                "quote": '"I know where it is," Finn says.',
                "explanation": "Explicit Finn dialogue tag.",
            },
        ]), JOB, CATALOG)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start_offset"], 0)
        self.assertEqual(events[1]["character_id"], "finn")
        self.assertFalse(events[1]["verified"])

    def test_unknown_identity_preserved(self):
        event = parse_response(reply([{
            "kind": "narration",
            "character_id": None,
            "quote": "Mara glances at the lantern.",
            "explanation": "Narrator describes an action.",
        }]), JOB, CATALOG)[0]

        self.assertIsNone(event["character_id"])

    def test_narration_cannot_claim_character_as_speaker(self):
        with self.assertRaisesRegex(
            ValueError, "Narration cannot"
        ):
            parse_response(reply([{
                "kind": "narration",
                "character_id": "finn",
                "quote": "Mara glances at the lantern.",
                "explanation": "Narrator describes Mara.",
            }]), JOB, CATALOG)

    def test_narration_keeps_character_speaker_null(self):
        result = parse_response(reply([{
            "kind": "narration",
            "character_id": None,
            "quote": "Mara glances at the lantern.",
            "explanation": "A narrator describes Mara.",
        }]), JOB, CATALOG)

        self.assertIsNone(result[0]["character_id"])
        self.assertFalse(result[0]["verified"])

    def test_fabricated_quote_rejected(self):
        with self.assertRaisesRegex(ValueError, "Evidence"):
            parse_response(reply([{
                "kind": "speech",
                "character_id": "finn",
                "quote": "The secret is under the table.",
                "explanation": "Invented text.",
            }]), JOB, CATALOG)

    def test_unknown_character_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown character"):
            parse_response(reply([{
                "kind": "speech",
                "character_id": "invented-person",
                "quote": '"I know where it is,"',
                "explanation": "Unlisted identity.",
            }]), JOB, CATALOG)

    def test_repeated_quote_has_no_unique_offset(self):
        job = {"turn": 1, "text": "Wait. Wait."}
        event = parse_response(reply([{
            "kind": "speech",
            "character_id": None,
            "quote": "Wait.",
            "explanation": "Speaker unclear.",
        }]), job, CATALOG)[0]

        self.assertIsNone(event["start_offset"])
        self.assertEqual(event["evidence_occurrences"], 2)

    def test_catalog_from_manifest(self):
        catalog = character_catalog({
            "character_cards": [
                {"card": {"id": "finn", "name": "Finn"}},
                {"card": {"id": "mara-vey", "name": "Mara Vey"}},
            ]
        })
        self.assertEqual(len(catalog), 2)

    def test_offline_resume_preserves_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )
            originals = {
                p.name: p.read_bytes()
                for p in archive.iterdir()
                if p.is_file()
            }

            calls = []

            def mock_chat(config, payload):
                calls.append(payload)
                return reply([])

            with patch(
                "rpbench.attribution.request_chat",
                side_effect=mock_chat,
            ):
                root, partial = evaluate(
                    archive, CONFIG, max_jobs=1
                )
                self.assertEqual(partial["status"], "partial")

                _, complete = evaluate(archive, CONFIG)
                evaluate(archive, CONFIG)

            self.assertEqual(complete["status"], "complete")
            self.assertEqual(
                len(calls), complete["total_jobs"]
            )
            self.assertTrue(
                (root / "attribution_events.jsonl").exists()
            )

            for name, original in originals.items():
                self.assertEqual(
                    (archive / name).read_bytes(),
                    original,
                )


if __name__ == "__main__":
    unittest.main()
