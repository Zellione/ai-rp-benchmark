import json
import tempfile
import unittest
from pathlib import Path

from rpbench.review import (
    propose,
    decide,
    rebuild,
    proposals,
    decisions,
)
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


class ReviewTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.run = run_simulation(
            ROOT / "scenarios/blackwood-selfplay.en.json",
            ROOT / "players/elena-ward.en.json",
            ROOT / "configs/mock.json",
            self.temp.name,
            turns=3,
        )

    def observation(self, turn=1, role="npc"):
        path = self.run / "observations.jsonl"
        entries = [
            json.loads(line)
            for line in path.read_text().splitlines()
        ]
        return next(
            o["text"] for o in entries
            if o["turn"] == turn and o["role"] == role
        )

    def test_proposal_requires_real_evidence(self):
        with self.assertRaises(ValueError):
            propose(
                self.run, "persona_violation",
                1, "npc",
                "This text was never generated",
                "Unsupported persona shift",
            )

        self.assertEqual(proposals(self.run), [])

    def test_rejected_proposal_changes_no_state(self):
        item = propose(
            self.run, "lore_error",
            1, "npc", self.observation(),
            "Potential lore error",
        )
        decide(
            self.run, item["id"],
            "reject", "human-test",
            "Not enough evidence",
        )

        state, evaluations = rebuild(self.run)
        self.assertEqual(
            state["last_confirmed_event"], 0
        )
        self.assertEqual(evaluations, [])
        self.assertEqual(len(decisions(self.run)), 1)

    def test_accepted_evaluation(self):
        item = propose(
            self.run, "persona_violation",
            1, "npc", self.observation(),
            "Test evaluation",
        )
        decide(
            self.run, item["id"],
            "accept", "human-test",
        )

        state, evaluations = rebuild(self.run)

        self.assertEqual(
            state["last_confirmed_event"], 0
        )
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(
            evaluations[0]["type"],
            "persona_violation",
        )

    def test_accepted_goal_state_event(self):
        item = propose(
            self.run, "goal_status_changed",
            1, "npc", self.observation(),
            "Goal completion confirmed for test",
            {
                "goal_id": "subgoal-01",
                "status": "completed",
            },
        )
        decide(
            self.run, item["id"],
            "accept", "human-test",
        )

        state, _ = rebuild(self.run)
        self.assertEqual(
            state["goals"]["subgoal-01"]["status"],
            "completed",
        )
        self.assertEqual(
            state["last_confirmed_event"], 1
        )

    def test_unknown_goal_cannot_be_accepted(self):
        item = propose(
            self.run, "goal_status_changed",
            1, "npc", self.observation(),
            "Invalid goal for validation test",
            {
                "goal_id": "unknown-goal",
                "status": "completed",
            },
        )

        with self.assertRaises(ValueError):
            decide(
                self.run, item["id"],
                "accept", "human-test",
            )

        self.assertEqual(decisions(self.run), [])

    def test_duplicate_decision_rejected(self):
        item = propose(
            self.run, "refusal",
            1, "npc", self.observation(),
            "Possible out-of-character refusal",
        )
        decide(
            self.run, item["id"],
            "reject", "human-test",
        )

        with self.assertRaises(ValueError):
            decide(
                self.run, item["id"],
                "accept", "human-test",
            )


if __name__ == "__main__":
    unittest.main()
