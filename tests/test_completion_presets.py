import tempfile
import unittest
from pathlib import Path

from rpbench.completion_presets import resolve_completion_preset
from rpbench.core import Backend, dump, load
from rpbench.observations import read_observations
from rpbench.resume import _histories, resume_simulation
from rpbench.simulation import run_simulation


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/character-chemistry-de/scenario.json"
PLAYER = ROOT / "scenarios/character-chemistry-de/player.json"
PRESETS = ROOT / "configs/completion-presets"


class CompletionPresetTests(unittest.TestCase):

    def test_named_speakers_both_perspectives(self):
        player = load(PLAYER)
        cards = [(Path("elara.json"), {"name": "Elara Wynn"})]

        config = {
            "backend": {"name": "llamacpp", "mode": "completion"},
            "completion_preset": str(
                PRESETS / "collaborative-raw-v1.json"
            ),
        }

        npc_fmt, player_fmt, info = resolve_completion_preset(
            config, ROOT / "configs/test.json", player, cards
        )

        backend = Backend(
            name="llamacpp",
            base_url="http://127.0.0.1:9999",
            model="test",
            mode="completion",
        )

        npc_history = [
            {"role": "system", "content": "Elara rules"},
            {"role": "user", "content": '"Hallo, Elara."'},
        ]

        player_history = [
            {"role": "system", "content": "Rowan rules"},
            {"role": "user", "content": "Rowan enters."},
            {"role": "assistant", "content": '"Elara?"'},
            {"role": "user", "content": '"Was willst du?"'},
        ]

        _, npc_payload = backend.prepare(npc_history, {}, npc_fmt)
        _, player_payload = backend.prepare(
            player_history, {}, player_fmt
        )

        self.assertTrue(
            npc_payload["prompt"].endswith("Elara Wynn:\n")
        )
        self.assertIn('Rowan Vale:\n"Hallo, Elara."', npc_payload["prompt"])

        self.assertTrue(
            player_payload["prompt"].endswith("Rowan Vale:\n")
        )
        self.assertIn(
            "Scene:\nRowan enters.",
            player_payload["prompt"],
        )
        self.assertIn(
            'Elara Wynn:\n"Was willst du?"',
            player_payload["prompt"],
        )
        self.assertEqual(info["id"], "collaborative-raw-v1")
        self.assertEqual(len(info["sha256"]), 64)

    def test_presets_preserve_speaker_order_after_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"

            dump(config_path, {
                "backend": {
                    "name": "mock",
                    "mode": "completion",
                    "model": "mock",
                },
                "sampling": {"max_tokens": 128},
                "completion_preset": str(
                    PRESETS / "rp-dialogue-v1.json"
                ),
            })

            run = run_simulation(
                SCENARIO, PLAYER, config_path,
                root / "runs", turns=2,
            )

            original_manifest = load(run / "manifest.json")
            original_formats = (
                original_manifest["prompt_format"],
                original_manifest["player_prompt_format"],
            )

            resume_simulation(run, turns=4)
            manifest = load(run / "manifest.json")
            entries = read_observations(run)

            self.assertEqual(
                original_formats,
                (
                    manifest["prompt_format"],
                    manifest["player_prompt_format"],
                ),
            )
            self.assertEqual(len(entries), 8)

            player_history, npc_history = _histories(
                manifest, entries
            )
            backend = Backend(
                name="llamacpp",
                base_url="http://127.0.0.1:9999",
                model="test",
                mode="completion",
            )

            _, player_payload = backend.prepare(
                player_history,
                {},
                manifest["player_prompt_format"],
            )
            _, npc_payload = backend.prepare(
                npc_history,
                {},
                manifest["prompt_format"],
            )

            self.assertTrue(
                player_payload["prompt"].endswith("Rowan Vale:\n")
            )
            self.assertTrue(
                npc_payload["prompt"].endswith("Elara Wynn:\n")
            )
            self.assertEqual(
                player_payload["prompt"].count("Opening scene:\n"),
                1,
            )
            self.assertEqual(
                npc_payload["prompt"].count("Rowan Vale:\n"),
                4,
            )


    def test_prompt_prefix_survives_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"

            dump(config_path, {
                "backend": {
                    "name": "mock",
                    "mode": "completion",
                    "model": "mock",
                },
                "sampling": {"max_tokens": 128},
                "completion_preset": str(
                    PRESETS / "rp-dialogue-v1.json"
                ),
            })

            run = run_simulation(
                SCENARIO, PLAYER, config_path,
                root / "runs", turns=2,
            )

            before_manifest = load(run / "manifest.json")
            before_entries = read_observations(run)
            before_player, before_npc = _histories(
                before_manifest, before_entries
            )

            backend = Backend(
                name="llamacpp",
                base_url="http://127.0.0.1:9999",
                model="test",
                mode="completion",
            )

            _, before_player_payload = backend.prepare(
                before_player,
                {},
                before_manifest["player_prompt_format"],
            )
            _, before_npc_payload = backend.prepare(
                before_npc,
                {},
                before_manifest["prompt_format"],
            )

            resume_simulation(run, turns=4)

            after_manifest = load(run / "manifest.json")
            after_entries = read_observations(run)
            after_player, after_npc = _histories(
                after_manifest, after_entries
            )

            _, after_player_payload = backend.prepare(
                after_player,
                {},
                after_manifest["player_prompt_format"],
            )
            _, after_npc_payload = backend.prepare(
                after_npc,
                {},
                after_manifest["prompt_format"],
            )

            for before, after, speaker in (
                (
                    before_player_payload["prompt"],
                    after_player_payload["prompt"],
                    "Rowan Vale:",
                ),
                (
                    before_npc_payload["prompt"],
                    after_npc_payload["prompt"],
                    "Elara Wynn:",
                ),
            ):
                # Drop the open speaker prefix at the end
                # of the shorter prompt. Earlier dialogue
                # must remain an exact prefix.
                self.assertTrue(before.rstrip().endswith(speaker))
                prefix = before[:before.rfind(speaker)]
                self.assertTrue(after.startswith(prefix))

            self.assertEqual(
                after_manifest["prompt_format"],
                before_manifest["prompt_format"],
            )
            self.assertEqual(
                after_manifest["player_prompt_format"],
                before_manifest["player_prompt_format"],
            )

    def test_chat_mode_rejects_completion_preset(self):
        with self.assertRaisesRegex(
            ValueError, "requires completion mode"
        ):
            resolve_completion_preset(
                {
                    "backend": {"name": "mock", "mode": "chat"},
                    "completion_preset": str(
                        PRESETS / "rp-dialogue-v1.json"
                    ),
                },
                ROOT / "configs/test.json",
                load(PLAYER),
                [(Path("elara.json"), {"name": "Elara Wynn"})],
            )


if __name__ == "__main__":
    unittest.main()
