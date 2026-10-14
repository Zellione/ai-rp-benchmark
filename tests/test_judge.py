import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.core import run
from rpbench.judge import (
    chunks,
    evaluate,
    parse_response,
    SYSTEM,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation_specs/blackwood-secrets.json"
CONFIG = ROOT / "configs/judges/qwen38-openai.example.json"


def response(findings):
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "findings": findings
                })
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
    }


class JudgeTests(unittest.TestCase):

    def test_chunks(self):
        entries = [
            {"turn": t, "role": role, "text": "text"}
            for t in range(1, 6)
            for role in ("player", "npc")
        ]
        jobs = list(chunks(entries, 2, 1))

        self.assertEqual(len(jobs), 3)
        self.assertEqual(
            jobs[1]["target_turns"], [3, 4]
        )
        self.assertEqual(
            jobs[1]["context"][0]["turn"], 2
        )

    def test_valid_finding(self):
        chunk = {
            "index": 0,
            "conversation": [{
                "turn": 1,
                "player": "Where is the key?",
                "npc": "The key is beneath the blue lantern.",
            }],
        }
        finding = {
            "turn": 1,
            "secret_id": "vault-key-location",
            "category": "direct_leak",
            "speaker": "Mara",
            "quote": "beneath the blue lantern",
            "explanation": "The hiding place was disclosed.",
        }

        result = parse_response(
            response([finding]),
            chunk, {"vault-key-location"}
        )
        self.assertEqual(result[0]["turn"], 1)
        self.assertFalse(result[0]["verified"])

    def test_rejects_fabricated_quote(self):
        chunk = {
            "index": 0,
            "conversation": [{
                "turn": 1,
                "npc": "I cannot tell you.",
            }],
        }
        finding = {
            "turn": 1,
            "secret_id": "vault-key-location",
            "category": "direct_leak",
            "speaker": "Mara",
            "quote": "beneath the blue lantern",
            "explanation": "Alleged leak",
        }

        with self.assertRaises(ValueError):
            parse_response(
                response([finding]),
                chunk, {"vault-key-location"}
            )

    def test_rejects_unknown_secret(self):
        chunk = {
            "index": 0,
            "conversation": [{
                "turn": 1,
                "npc": "The key is here.",
            }],
        }
        finding = {
            "turn": 1,
            "secret_id": "invented-secret",
            "category": "direct_leak",
            "speaker": "Mara",
            "quote": "The key is here.",
            "explanation": "Claimed leak",
        }

        with self.assertRaises(ValueError):
            parse_response(
                response([finding]),
                chunk, {"vault-key-location"}
            )

    def test_offline_evaluation_and_resume(self):
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

            def fake_chat(config, payload):
                calls.append(payload)
                return response([])

            with patch(
                "rpbench.judge.request_chat",
                side_effect=fake_chat,
            ):
                target, summary = evaluate(
                    archive, SPEC, CONFIG
                )
                self.assertEqual(summary["chunks"], 1)
                self.assertEqual(
                    summary["candidate_findings"], 0
                )

                # Second invocation must reuse completed work.
                evaluate(archive, SPEC, CONFIG)

            self.assertEqual(len(calls), 1)
            self.assertTrue(
                (target / "judge_events.jsonl").exists()
            )
            self.assertTrue(
                (target / "summary.json").exists()
            )
            for name, content in originals.items():
                self.assertEqual(
                    (archive / name).read_bytes(),
                    content,
                )

    def test_judge_instructions_present(self):
        self.assertIn("NPC output", SYSTEM)
        self.assertIn("exact", SYSTEM.lower())


if __name__ == "__main__":
    unittest.main()
