import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.core import run
from rpbench.judge_long import evaluate
from rpbench.long_context import fit_token_budget

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation_specs/blackwood-secrets.json"
CONFIG = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"


def fake_fit(job, make_payload, config, max_tokens):
    return fit_token_budget(
        job,
        make_payload,
        config,
        max_tokens,
        counter=lambda payload, cfg: 100,
    )


def fake_response(actual_tokens):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": json.dumps({
                    "disclosure": "none",
                    "knowledge_violation": False,
                    "speaker": "unknown",
                    "quote": "",
                    "explanation": "No disclosure.",
                })
            },
        }],
        "usage": {
            "prompt_tokens": actual_tokens,
            "completion_tokens": 12,
        },
    }


class TokenVerificationTests(unittest.TestCase):

    def test_verified_token_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )

            with patch(
                "rpbench.judge_long.fit_token_budget",
                side_effect=fake_fit,
            ), patch(
                "rpbench.judge_long.request_chat",
                return_value=fake_response(100),
            ):
                root, summary = evaluate(
                    archive, SPEC, CONFIG,
                    max_jobs=1,
                    max_prompt_tokens=8192,
                    max_prompt_bytes=32768,
                )

            self.assertEqual(summary["completed_jobs"], 1)

            events = [
                json.loads(line)
                for line in (
                    root / "judge_events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            check = next(
                e for e in events
                if e["type"] == "token_count_verification"
            )

            self.assertTrue(check["matches"])
            self.assertEqual(check["difference"], 0)

            request = next(
                e for e in events if e["type"] == "request"
            )
            audit = request["context_selection"]["budget_audit"]

            size = len(json.dumps(
                request["payload"], ensure_ascii=False
            ).encode("utf-8"))

            self.assertEqual(audit["final_bytes"], size)

    def test_token_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )

            with patch(
                "rpbench.judge_long.fit_token_budget",
                side_effect=fake_fit,
            ), patch(
                "rpbench.judge_long.request_chat",
                return_value=fake_response(101),
            ):
                with self.assertRaisesRegex(
                    ValueError, "token counts differ"
                ):
                    evaluate(
                        archive, SPEC, CONFIG,
                        max_jobs=1,
                        max_prompt_tokens=8192,
                    )

            roots = list(
                (archive / "evaluations").glob(
                    "semantic-secrets-long-v1-*"
                )
            )
            self.assertEqual(len(roots), 1)

            events = [
                json.loads(line)
                for line in (
                    roots[0] / "judge_events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            self.assertFalse(any(
                e["type"] == "job_complete"
                for e in events
            ))
            self.assertTrue(any(
                e["type"] == "token_count_verification"
                and not e["matches"]
                for e in events
            ))


if __name__ == "__main__":
    unittest.main()
