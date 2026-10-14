import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.core import run
from rpbench.judge_v2 import (
    evaluate,
    make_jobs,
    parse_response,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation_specs/blackwood-secrets.json"
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"


def response(disclosure="none", knowledge=False,
             quote="", speaker="Mara"):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": json.dumps({
                    "disclosure": disclosure,
                    "knowledge_violation": knowledge,
                    "speaker": speaker,
                    "quote": quote,
                    "explanation": "Test judgment.",
                })
            },
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
        },
    }


class JudgeV2Tests(unittest.TestCase):

    def setUp(self):
        self.spec = json.loads(
            SPEC.read_text(encoding="utf-8")
        )
        self.entries = [
            {"turn": 1, "role": "player", "text": "Where?"},
            {"turn": 1, "role": "npc", "text": "No."},
            {"turn": 2, "role": "player", "text": "Why?"},
            {"turn": 2, "role": "npc", "text": "I refuse."},
        ]

    def test_jobs_have_full_prior_context(self):
        jobs = list(make_jobs(self.entries, self.spec))
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["prior_context"], [])
        self.assertEqual(
            jobs[1]["prior_context"][0]["npc"], "No."
        )

    def test_two_independent_dimensions(self):
        job = list(make_jobs(self.entries, self.spec))[0]
        found = parse_response(
            response(
                disclosure="direct",
                knowledge=True,
                quote="No.",
                speaker="Finn",
            ),
            job,
        )
        self.assertEqual(found["disclosure"], "direct")
        self.assertTrue(found["knowledge_violation"])
        self.assertFalse(found["speaker_verified"])

    def test_fabricated_quote_rejected(self):
        job = list(make_jobs(self.entries, self.spec))[0]
        with self.assertRaises(ValueError):
            parse_response(
                response(
                    disclosure="direct",
                    quote="invented quote",
                ),
                job,
            )

    def test_positive_judgment_needs_evidence(self):
        job = list(make_jobs(self.entries, self.spec))[0]
        with self.assertRaises(ValueError):
            parse_response(
                response(
                    knowledge=True,
                    quote="",
                ),
                job,
            )

    def test_partial_run_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                temp,
            )
            originals = {
                p.name: p.read_bytes()
                for p in archive.iterdir()
                if p.is_file()
            }

            calls = []

            def fake_chat(config, payload):
                calls.append(payload)
                return response()

            with patch(
                "rpbench.judge_v2.request_chat",
                side_effect=fake_chat,
            ):
                root, partial = evaluate(
                    archive, SPEC, CONFIG, max_jobs=1
                )
                self.assertEqual(partial["status"], "partial")
                self.assertEqual(partial["completed_jobs"], 1)

                root2, complete = evaluate(
                    archive, SPEC, CONFIG
                )
                self.assertEqual(root, root2)
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(
                    complete["completed_jobs"],
                    complete["total_jobs"],
                )

                evaluate(archive, SPEC, CONFIG)

            self.assertEqual(
                len(calls), complete["total_jobs"]
            )
            self.assertEqual(
                len((root / "findings.jsonl").read_text()
                    .splitlines()),
                complete["total_jobs"],
            )

            for name, content in originals.items():
                self.assertEqual(
                    (archive / name).read_bytes(),
                    content,
                )

    def test_truncation_rejected(self):
        job = list(make_jobs(self.entries, self.spec))[0]
        result = response()
        result["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            parse_response(result, job)


if __name__ == "__main__":
    unittest.main()
