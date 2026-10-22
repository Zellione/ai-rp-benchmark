import tempfile
import unittest
from pathlib import Path

from rpbench.core import dump, load, raw_prompt
from rpbench.observations import read_observations
from rpbench.resume import resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = (
    ROOT / "scenarios/character-chemistry-de/scenario.json"
)
PLAYER = (
    ROOT / "scenarios/character-chemistry-de/player.json"
)


class SelfplayCompletionTests(unittest.TestCase):

    def test_raw_prompt_preserves_role_order(self):
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Guten Abend"},
        ]

        result = raw_prompt(messages, {})

        self.assertLess(
            result.index("System: Rules"),
            result.index("User: Hallo"),
        )
        self.assertLess(
            result.index("User: Hallo"),
            result.index("Assistant: Guten Abend"),
        )
        self.assertTrue(result.endswith("Assistant: "))

    def test_completion_selfplay_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "mock-completion.json"

            dump(config_path, {
                "backend": {
                    "name": "mock",
                    "mode": "completion",
                    "model": "mock",
                },
                "sampling": {
                    "max_tokens": 256,
                },
                "prompt_format": {
                    "prefixes": {
                        "system": "Regeln: ",
                        "user": "Spieler: ",
                        "assistant": "Charakter: ",
                    },
                    "separator": "\n\n",
                    "assistant_prefix": "Charakter: ",
                },
            })

            run = run_simulation(
                SCENARIO,
                PLAYER,
                config_path,
                root / "runs",
                turns=2,
            )

            manifest = load(run / "manifest.json")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_turns"], 2)

            resumed = resume_simulation(run, turns=4)
            self.assertEqual(resumed, run.resolve())

            final = load(run / "manifest.json")
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["completed_turns"], 4)

            observations = read_observations(run)
            self.assertEqual(len(observations), 8)

            text = (run / "conversation.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(text.count("### NPC"), 4)


if __name__ == "__main__":
    unittest.main()
