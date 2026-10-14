import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rpbench.simulation import (
    create_player_prompt,
    make_histories,
    run_simulation,
)

ROOT = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    def test_player_prompt_contains_goals(self):
        player = {
            "name": "Elena",
            "persona": "Historian",
            "main_goal": "Investigate the archive",
            "subgoals": ["Gain access", "Read records"],
        }
        prompt = create_player_prompt(player)
        self.assertIn("Investigate the archive", prompt)
        self.assertIn("Gain access", prompt)

    def test_system_prompt_isolation(self):
        player_messages, npc_messages = make_histories(
            "PLAYER PRIVATE GOALS",
            "NPC SECRET BLUE LANTERN",
            "Enter the archive",
        )

        self.assertNotIn(
            "NPC SECRET BLUE LANTERN",
            json.dumps(player_messages),
        )
        self.assertNotIn(
            "PLAYER PRIVATE GOALS",
            json.dumps(npc_messages),
        )

    def test_six_turn_mock_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_simulation(
                ROOT / "scenarios/blackwood-selfplay.en.json",
                ROOT / "players/elena-ward.en.json",
                ROOT / "configs/mock.json",
                tmp,
                turns=6,
            )

            manifest = json.loads(
                (run_dir / "manifest.json").read_text()
            )
            events = [
                json.loads(line)
                for line in (
                    run_dir / "events.jsonl"
                ).read_text().splitlines()
            ]

            transcript = (
                run_dir / "conversation.md"
            ).read_text()

            self.assertEqual(
                manifest["status"], "complete"
            )
            self.assertEqual(
                manifest["completed_turns"], 6
            )
            self.assertEqual(len(events), 24)
            self.assertEqual(
                transcript.count("## Turn"), 6
            )

    def test_real_request_isolation(self):
        captured = []

        def fake_generate(backend, url, payload):
            captured.append(payload)
            return "A short response.", {
                "choices": [{
                    "message": {
                        "content": "A short response.",
                        "reasoning": None,
                    },
                    "finish_reason": "stop",
                }]
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "rpbench.simulation.Backend.generate",
                fake_generate,
            ):
                run_simulation(
                    ROOT / "scenarios/blackwood-selfplay.en.json",
                    ROOT / "players/elena-ward.en.json",
                    ROOT / "configs/llamacpp-chat.example.json",
                    tmp,
                    turns=2,
                )

        self.assertEqual(len(captured), 4)

        player_requests = captured[0::2]
        npc_requests = captured[1::2]

        for payload in player_requests:
            rendered = json.dumps(payload)
            self.assertNotIn(
                "blue lantern", rendered.lower()
            )

        for payload in npc_requests:
            rendered = json.dumps(payload)
            self.assertNotIn(
                "Discover what really happened",
                rendered,
            )


if __name__ == "__main__":
    unittest.main()
