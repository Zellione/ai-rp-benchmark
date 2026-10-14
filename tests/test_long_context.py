import json
import unittest
from pathlib import Path

from rpbench.long_context import select_context
from rpbench.judge_long import make_jobs, make_payload

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads(
    (ROOT / "evaluation_specs/blackwood-secrets.json")
    .read_text(encoding="utf-8")
)
CONFIG = json.loads(
    (ROOT / "configs/judges/qwen38-27b-ninfer-32k.json")
    .read_text(encoding="utf-8")
)


def history(n=400):
    return [
        {
            "turn": i,
            "player": f"Ordinary conversation {i}.",
            "npc": f"Ordinary response {i}.",
        }
        for i in range(1, n + 1)
    ]


class LongContextTests(unittest.TestCase):

    def test_bounded_400_turn_history(self):
        data = history()
        selection = select_context(
            data, SPEC["secrets"][0],
            window_turns=12,
            evidence_turns=8,
        )
        self.assertLessEqual(
            len(selection["recent_context"])
            + len(selection["retrieved_older_evidence"]),
            20,
        )
        self.assertEqual(
            selection["coverage"]["total_prior_turns"], 400
        )
        self.assertFalse(
            selection["coverage"]["history_complete"]
        )

    def test_old_secret_evidence_retrieved(self):
        data = history()
        data[29]["npc"] = (
            "Mara told Finn that the vault key is hidden "
            "beneath the blue lantern in her office."
        )

        selection = select_context(
            data, SPEC["secrets"][0],
            window_turns=12,
            evidence_turns=8,
        )

        selected = [
            t["turn"] for t in
            selection["retrieved_older_evidence"]
        ]
        self.assertIn(30, selected)
        self.assertEqual(
            selection["retrieved_older_evidence"][0]["npc"],
            data[29]["npc"],
        )

    def test_no_future_leakage(self):
        entries = []
        for turn in range(1, 5):
            entries.extend([
                {
                    "turn": turn,
                    "role": "player",
                    "text": f"Player {turn}",
                },
                {
                    "turn": turn,
                    "role": "npc",
                    "text": (
                        "The vault key is under the blue lantern."
                        if turn == 4 else f"NPC {turn}"
                    ),
                },
            ])

        jobs = list(make_jobs(
            entries, SPEC,
            window_turns=2,
            evidence_turns=1,
        ))
        self.assertEqual(len(jobs), 4)
        for job in jobs[:3]:
            context = job["context_selection"]
            texts = json.dumps(context)
            self.assertNotIn(
                "under the blue lantern", texts
            )

    def test_payload_has_coverage(self):
        entries = []
        for turn in range(1, 31):
            entries.extend([
                {
                    "turn": turn,
                    "role": "player",
                    "text": f"Player {turn}",
                },
                {
                    "turn": turn,
                    "role": "npc",
                    "text": f"NPC {turn}",
                },
            ])

        job = list(make_jobs(
            entries, SPEC,
            window_turns=4,
            evidence_turns=2,
        ))[-1]
        payload = make_payload(job, CONFIG)
        task = json.loads(payload["messages"][1]["content"])

        self.assertEqual(
            task["context_coverage"]["total_prior_turns"],
            29,
        )
        self.assertLessEqual(
            len(task["prior_context"])
            + len(task["retrieved_older_evidence"]),
            6,
        )

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            select_context(
                history(3), SPEC["secrets"][0],
                window_turns=0,
            )


if __name__ == "__main__":
    unittest.main()
