import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.core import run
from rpbench.knowledge_transfers import (
    evaluate,
    parse_response,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"

CATALOG = [
    {"id": "mara-vey", "name": "Mara Vey"},
    {"id": "finn-calder", "name": "Finn Calder"},
]

JOB = {
    "turn": 3,
    "text": (
        "Mara tells Finn: The key is beneath the blue lantern."
    ),
}


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
    }


def transfer(kind="explicit_transfer"):
    return {
        "kind": kind,
        "source_character_id": "mara-vey",
        "recipient_character_id": "finn-calder",
        "fact": "The key is beneath the blue lantern.",
        "recipient_prior_knowledge": "unknown",
        "quote": "The key is beneath the blue lantern.",
        "explanation": "Mara tells Finn the location.",
    }


class KnowledgeTransferTests(unittest.TestCase):

    def test_explicit_transfer_candidate(self):
        result = parse_response(
            response([transfer()]), JOB, CATALOG
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["source_character_id"],
            "mara-vey",
        )
        self.assertFalse(
            result[0]["acquisition_confirmed"]
        )
        self.assertFalse(result[0]["verified"])

    def test_reported_transfer_candidate(self):
        result = parse_response(
            response([transfer("reported_transfer")]),
            JOB, CATALOG,
        )
        self.assertEqual(
            result[0]["kind"], "reported_transfer"
        )

    def test_prior_knowledge_requires_valid_status(self):
        event = {
            **transfer(),
            "recipient_prior_knowledge": "assumed_new",
        }
        with self.assertRaisesRegex(
            ValueError, "prior knowledge"
        ):
            parse_response(
                response([event]), JOB, CATALOG
            )

    def test_unknown_prior_knowledge_is_preserved(self):
        result = parse_response(
            response([transfer()]), JOB, CATALOG
        )
        self.assertEqual(
            result[0]["recipient_prior_knowledge"],
            "unknown",
        )
        self.assertFalse(
            result[0]["acquisition_confirmed"]
        )

    def test_unknown_source_rejected(self):
        event = {
            **transfer(),
            "source_character_id": "invented",
        }
        with self.assertRaisesRegex(
            ValueError, "Unknown source"
        ):
            parse_response(response([event]), JOB, CATALOG)

    def test_invented_quote_rejected(self):
        event = {
            **transfer(),
            "quote": "A secret never mentioned.",
        }
        with self.assertRaisesRegex(
            ValueError, "missing from NPC"
        ):
            parse_response(response([event]), JOB, CATALOG)

    def test_same_source_recipient_rejected(self):
        event = {
            **transfer(),
            "recipient_character_id": "mara-vey",
        }
        with self.assertRaisesRegex(
            ValueError, "distinct characters"
        ):
            parse_response(response([event]), JOB, CATALOG)

    def test_resume_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )

            calls = []

            def mock_chat(config, payload):
                calls.append(payload)
                return response([])

            with patch(
                "rpbench.knowledge_transfers.request_chat",
                side_effect=mock_chat,
            ):
                _, partial = evaluate(
                    archive, CONFIG, max_jobs=1
                )
                self.assertEqual(
                    partial["status"], "partial"
                )
                _, complete = evaluate(archive, CONFIG)
                evaluate(archive, CONFIG)

            self.assertEqual(complete["status"], "complete")
            self.assertEqual(
                len(calls), complete["total_jobs"]
            )

    def test_truncation_rejected(self):
        raw = response([])
        raw["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(
            ValueError, "Truncated"
        ):
            parse_response(raw, JOB, CATALOG)


if __name__ == "__main__":
    unittest.main()
