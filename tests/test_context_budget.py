import json
import unittest
from pathlib import Path

from rpbench.judge_long import make_jobs, make_payload
from rpbench.long_context import fit_payload_budget


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads(
    (ROOT / "evaluation_specs/blackwood-secrets.json")
    .read_text(encoding="utf-8")
)
CONFIG = json.loads(
    (ROOT / "configs/judges/qwen38-27b-ninfer-32k.json")
    .read_text(encoding="utf-8")
)


class ContextBudgetTests(unittest.TestCase):

    def make_job(self):
        entries = []
        for turn in range(1, 31):
            entries.extend([
                {
                    "turn": turn,
                    "role": "player",
                    "text": "Question " + "x" * 800,
                },
                {
                    "turn": turn,
                    "role": "npc",
                    "text": (
                        "The vault key is near the blue lantern. "
                        + "z" * 800
                    ),
                },
            ])

        return list(make_jobs(
            entries, SPEC,
            window_turns=12,
            evidence_turns=8,
        ))[-1]

    def test_request_is_bounded(self):
        job = self.make_job()
        selected = fit_payload_budget(
            job, make_payload, CONFIG, 10000
        )
        payload = make_payload(selected, CONFIG)
        size = len(json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"))

        self.assertLessEqual(size, 10000)
        self.assertGreater(
            selected["context_selection"]
            ["coverage"]["omitted_prior_turns"],
            0,
        )
        self.assertFalse(
            selected["context_selection"]
            ["coverage"]["history_complete"]
        )
        self.assertFalse(
            selected["context_selection"]
            ["budget_audit"]["token_count_verified"]
        )
        self.assertEqual(
            selected["context_selection"]
            ["budget_audit"]["final_bytes"],
            size,
        )

    def test_source_job_is_unchanged(self):
        job = self.make_job()
        before = json.dumps(job, sort_keys=True)

        fit_payload_budget(
            job, make_payload, CONFIG, 10000
        )

        self.assertEqual(
            json.dumps(job, sort_keys=True),
            before,
        )

    def test_target_is_never_truncated(self):
        job = self.make_job()

        with self.assertRaises(ValueError):
            fit_payload_budget(
                job, make_payload, CONFIG, 100
            )

        self.assertEqual(
            len(job["target"]["npc"]),
            840,
        )

    def test_invalid_limits_rejected(self):
        job = self.make_job()
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    fit_payload_budget(
                        job, make_payload, CONFIG, value
                    )


if __name__ == "__main__":
    unittest.main()


class BudgetIntegrationTests(unittest.TestCase):

    def test_evaluation_uses_budget(self):
        import tempfile
        from unittest.mock import patch

        from rpbench.core import run
        from rpbench.judge_long import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )

            def fake_chat(config, payload):
                size = len(json.dumps(
                    payload, ensure_ascii=False
                ).encode("utf-8"))

                self.assertLessEqual(size, 32768)

                return {
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({
                                "disclosure": "none",
                                "knowledge_violation": False,
                                "speaker": "unknown",
                                "quote": "",
                                "explanation": "No secret disclosed.",
                            })
                        },
                    }],
                }

            with patch(
                "rpbench.judge_long.request_chat",
                side_effect=fake_chat,
            ):
                target, summary = evaluate(
                    archive,
                    ROOT / "evaluation_specs/blackwood-secrets.json",
                    ROOT / "configs/judges/qwen38-27b-ninfer-32k.json",
                    max_jobs=1,
                    max_prompt_bytes=32768,
                )

            events = [
                json.loads(line)
                for line in (
                    target / "judge_events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            request = next(
                e for e in events if e["type"] == "request"
            )

            audit = request["context_selection"]["budget_audit"]

            self.assertEqual(audit["max_bytes"], 32768)
            self.assertLessEqual(audit["final_bytes"], 32768)
            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(
                summary["context_policy"]["max_prompt_bytes"],
                32768,
            )
