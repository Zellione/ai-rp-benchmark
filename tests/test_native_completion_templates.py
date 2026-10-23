import unittest
from pathlib import Path

from rpbench.completion_presets import resolve_completion_preset
from rpbench.core import Backend, load, native_turn_prompt


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs/completion-presets"
PLAYER = ROOT / "scenarios/character-chemistry-de/player.json"


class NativeCompletionTemplateTests(unittest.TestCase):

    def resolve(self, filename):
        config = {
            "backend": {
                "name": "llamacpp",
                "mode": "completion",
            },
            "completion_preset": str(PRESETS / filename),
        }
        return resolve_completion_preset(
            config,
            ROOT / "configs/test.json",
            load(PLAYER),
            [(Path("elara.json"), {"name": "Elara Wynn"})],
        )

    def backend(self):
        return Backend(
            name="llamacpp",
            base_url="http://127.0.0.1:9999",
            model="test",
            mode="completion",
        )

    def test_gemma_system_in_first_user_turn(self):
        npc_fmt, player_fmt, _ = self.resolve(
            "gemma1-3-native-v1.json"
        )
        messages = [
            {"role": "system", "content": "Play Elara."},
            {"role": "user", "content": "Hallo."},
        ]
        rendered = native_turn_prompt(messages, npc_fmt)

        self.assertEqual(
            rendered,
            "<start_of_turn>user\n"
            "Play Elara.\n\nHallo.<end_of_turn>\n"
            "<start_of_turn>model\n",
        )
        self.assertEqual(npc_fmt, player_fmt)

    def test_muse_native_turn_boundaries(self):
        npc_fmt, _, _ = self.resolve(
            "muse-native-candidate-v1.json"
        )
        rendered = native_turn_prompt(
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Hallo"},
                {"role": "assistant", "content": "Guten Abend"},
                {"role": "user", "content": "Wie geht es dir?"},
            ],
            npc_fmt,
        )

        self.assertTrue(rendered.startswith(
            '<|start|>system<|message|>Rules'
            '\n\nReasoning strength: high.'
            '\n\n# Valid recipients: "self", "user".<|eot|>'
        ))
        self.assertTrue(rendered.endswith(
            "<|start|>assistant"
        ))
        self.assertEqual(rendered.count("<|start|>user"), 2)

    def test_gemma_stop_reaches_completion_payload(self):
        fmt, _, _ = self.resolve("gemma1-3-native-v1.json")
        url, payload = self.backend().prepare(
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Hallo"},
            ],
            {"max_tokens": 4096},
            fmt,
        )

        self.assertTrue(url.endswith("/v1/completions"))
        self.assertIn("<end_of_turn>", payload["stop"])
        self.assertEqual(payload["max_tokens"], 4096)

    def test_muse_stop_reaches_completion_payload(self):
        fmt, _, _ = self.resolve(
            "muse-native-candidate-v1.json"
        )
        _, payload = self.backend().prepare(
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Hallo"},
            ],
            {"stop": ["CUSTOM_STOP"]},
            fmt,
        )

        self.assertEqual(
            payload["stop"],
            ["CUSTOM_STOP", "<|eot|>"],
        )


if __name__ == "__main__":
    unittest.main()
