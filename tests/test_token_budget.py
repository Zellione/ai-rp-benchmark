import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.core import run
from rpbench.judge_long import evaluate, make_jobs, make_payload
from rpbench.long_context import (
    count_ninfer_prompt_tokens,
    fit_token_budget,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "evaluation_specs/blackwood-secrets.json"
CONFIG_PATH = ROOT / "configs/judges/qwen38-27b-ninfer-32k.json"

SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def fake_counter(payload, config):
    task = json.loads(payload["messages"][1]["content"])
    selected = (
        task["prior_context"]
        + task["retrieved_older_evidence"]
    )
    return 100 + 50 * len(selected)


def make_long_job():
    entries = []
    for turn in range(1, 11):
        entries.extend([
            {
                "turn": turn,
                "role": "player",
                "text": f"Question {turn}",
            },
            {
                "turn": turn,
                "role": "npc",
                "text": (
                    "Mara revealed the vault key "
                    "beneath the blue lantern."
                    if turn == 2
                    else f"Ordinary NPC response {turn}"
                ),
            },
        ])

    return list(make_jobs(
        entries, SPEC,
        window_turns=3,
        evidence_turns=2,
    ))[-1]


class TokenBudgetTests(unittest.TestCase):

    def test_bounded_prompt(self):
        job = make_long_job()
        before = json.dumps(job, sort_keys=True)

        fitted = fit_token_budget(
            job,
            make_payload,
            CONFIG,
            200,
            counter=fake_counter,
        )

        audit = fitted["context_selection"][
            "token_budget_audit"
        ]

        self.assertLessEqual(
            audit["final_prompt_tokens"], 200
        )
        self.assertTrue(audit["removed_turns"])
        self.assertEqual(json.dumps(job, sort_keys=True), before)

    def test_relevant_older_evidence_survives(self):
        job = make_long_job()

        fitted = fit_token_budget(
            job,
            make_payload,
            CONFIG,
            300,
            counter=fake_counter,
        )

        retained = fitted["context_selection"][
            "token_budget_audit"
        ]["retained_turns"]

        self.assertIn(2, retained)

    def test_target_cannot_be_removed(self):
        job = make_long_job()

        def impossible_counter(payload, config):
            return 100000

        with self.assertRaisesRegex(
            ValueError, "exceed"
        ):
            fit_token_budget(
                job,
                make_payload,
                CONFIG,
                100,
                counter=impossible_counter,
            )

    def test_bad_counter_result_rejected(self):
        job = make_long_job()

        def invalid_counter(payload, config):
            raise ValueError("Invalid token count")

        with self.assertRaisesRegex(
            ValueError, "Invalid token count"
        ):
            fit_token_budget(
                job,
                make_payload,
                CONFIG,
                1000,
                counter=invalid_counter,
            )

    def test_context_reserve_checked_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = run(
                ROOT / "scenarios/suspicious-archivist.json",
                ROOT / "configs/mock.json",
                tmp,
            )

            with patch(
                "rpbench.judge_long.request_chat"
            ) as request, patch(
                "rpbench.long_context.count_ninfer_prompt_tokens"
            ) as counter:
                with self.assertRaisesRegex(
                    ValueError, "exceeds context"
                ):
                    evaluate(
                        archive,
                        SPEC_PATH,
                        CONFIG_PATH,
                        max_prompt_tokens=10000,
                        context_limit=20000,
                    )

                request.assert_not_called()
                counter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
