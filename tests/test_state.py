import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from rpbench.state import (
    build_ground_truth,
    initial_state,
    make_observation,
    apply_confirmed_event,
)
from rpbench.characters import load_character_card
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


class StateTests(unittest.TestCase):

    def setUp(self):
        self.player = json.loads(
            (ROOT / "players/elena-ward.en.json")
            .read_text(encoding="utf-8")
        )
        self.cards = [
            load_character_card(
                ROOT / f"characters/{name}.en.json"
            )
            for name in (
                "mara-vey",
                "finn-calder",
                "alden-voss",
            )
        ]
        self.truth = build_ground_truth(
            self.player, self.cards
        )
        self.state = initial_state(self.truth)

    def test_private_knowledge_separation(self):
        mara = self.state["character_knowledge"][
            "mara-vey"
        ]
        finn = self.state["character_knowledge"][
            "finn-calder"
        ]

        self.assertTrue(
            any("blue lantern" in fact for fact in
                mara["private_facts"])
        )
        self.assertFalse(
            any("blue lantern" in fact for fact in
                finn["known_facts"] + finn["private_facts"])
        )

    def test_observation_does_not_change_state(self):
        before = deepcopy(self.state)

        observation = make_observation(
            1, "player",
            "I have found the vault key.",
        )

        self.assertFalse(observation["verified"])
        self.assertEqual(self.state, before)

        with self.assertRaises(ValueError):
            apply_confirmed_event(
                self.state, observation
            )

    def test_verified_knowledge_transfer(self):
        event = {
            "event_id": 1,
            "type": "knowledge_acquired",
            "verified": True,
            "verified_by": "human",
            "character_id": "finn-calder",
            "fact": "Mara keeps the vault key beneath the blue lantern",
        }

        updated = apply_confirmed_event(
            self.state, event
        )

        self.assertEqual(
            self.state["last_confirmed_event"], 0
        )
        self.assertEqual(
            updated["last_confirmed_event"], 1
        )
        self.assertIn(
            event["fact"],
            updated["character_knowledge"][
                "finn-calder"
            ]["known_facts"],
        )

    def test_goal_requires_confirmation(self):
        event = {
            "event_id": 1,
            "type": "goal_status_changed",
            "verified": False,
            "verified_by": "human",
            "goal_id": "subgoal-01",
            "status": "completed",
        }

        with self.assertRaises(ValueError):
            apply_confirmed_event(
                self.state, event
            )

        event["verified"] = True
        updated = apply_confirmed_event(
            self.state, event
        )

        self.assertEqual(
            updated["goals"][
                "subgoal-01"
            ]["status"],
            "completed",
        )
        self.assertEqual(
            self.state["goals"][
                "subgoal-01"
            ]["status"],
            "pending",
        )

    def test_rejects_out_of_order_events(self):
        event = {
            "event_id": 3,
            "type": "goal_status_changed",
            "verified": True,
            "verified_by": "human",
            "goal_id": "subgoal-01",
            "status": "active",
        }

        with self.assertRaises(ValueError):
            apply_confirmed_event(
                self.state, event
            )

    def test_run_archives_ground_truth_and_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_simulation(
                ROOT / "scenarios/blackwood-selfplay.en.json",
                ROOT / "players/elena-ward.en.json",
                ROOT / "configs/mock.json",
                tmp,
                turns=3,
            )

            truth = json.loads(
                (run_dir / "ground_truth.json").read_text()
            )
            state = json.loads(
                (run_dir / "state.json").read_text()
            )
            observations = [
                json.loads(line)
                for line in (
                    run_dir / "observations.jsonl"
                ).read_text().splitlines()
            ]

            self.assertEqual(len(truth["characters"]), 3)
            self.assertEqual(len(observations), 6)
            self.assertEqual(
                state["last_confirmed_event"], 0
            )
            self.assertTrue(
                all(not e["verified"] for e in observations)
            )


if __name__ == "__main__":
    unittest.main()
