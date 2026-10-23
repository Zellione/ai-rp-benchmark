import tempfile
import unittest
from pathlib import Path

from rpbench.completion_presets import resolve_completion_preset
from rpbench.completion_response import (
    CompletionParseError,
    parse_atem_completion,
    parse_atem_direct_completion,
    parse_atem_thinking_prefill,
)
from rpbench.core import Backend, load
from rpbench.observations import read_jsonl
from rpbench.simulation import generate_role


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "configs/completion-presets"


class FakeMuseBackend:
    name = "llamacpp"
    mode = "completion"

    def __init__(self, response):
        self.response = response

    def prepare(self, messages, sampling, prompt_format):
        return "mock://muse", {"prompt_format": prompt_format}

    def generate(self, url, payload):
        return self.response, {
            "choices": [
                {"text": self.response, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 50},
        }


class MuseATEMVariantTests(unittest.TestCase):

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
            ROOT / "configs/example.json",
            load("scenarios/character-chemistry-de/player.json"),
            [(Path("elara.json"), {"name": "Elara Wynn"})],
        )[0]

    def test_distinct_generation_prefixes(self):
        messages = [
            {"role": "system", "content": "Du spielst Elara."},
            {"role": "user", "content": '"Rowan?"'},
        ]
        backend = Backend(
            "llamacpp",
            "http://127.0.0.1:8082",
            "muse-glimmer-30b",
            "completion",
        )

        expected = {
            "muse-atem-reasoning-v1.json":
                "<|start|>assistant",
            "muse-atem-direct-v1.json":
                "<|start|>assistant to=user<|message|>",
        }

        for filename, ending in expected.items():
            with self.subTest(filename=filename):
                fmt = self.resolve(filename)
                _, payload = backend.prepare(messages, {}, fmt)
                self.assertTrue(payload["prompt"].endswith(ending))
                self.assertEqual(payload["stop"], ["<|eot|>"])

    def test_thinking_prefill_channel_separation(self):
        raw = (
            "Reason carefully about Elara."
            "<|start|>assistant to=user<|message|>"
            '*Elara schaut auf.* "Rowan?"'
        )
        visible, reasoning = parse_atem_thinking_prefill(raw)

        self.assertEqual(
            visible,
            '*Elara schaut auf.* "Rowan?"',
        )
        self.assertEqual(
            reasoning,
            "Reason carefully about Elara.",
        )

    def test_thinking_prefill_rejects_missing_visible(self):
        with self.assertRaises(CompletionParseError):
            parse_atem_thinking_prefill(
                "Only reasoning without a final answer"
            )

    def test_stray_atem_marker_inside_thinking(self):
        raw = (
            "First reasoning sentence."
            "<|start|>enough. Continue thinking."
            "<|start|>assistant to=user<|message|>"
            '*Elara schaut auf.* "Rowan?"'
        )

        visible, reasoning = parse_atem_thinking_prefill(raw)

        self.assertEqual(
            visible,
            '*Elara schaut auf.* "Rowan?"',
        )
        self.assertEqual(
            reasoning,
            "First reasoning sentence."
            "enough. Continue thinking.",
        )

    def test_stray_marker_with_standard_atem(self):
        raw = (
            " to=self<|message|>"
            "Reasoning.<|start|>Continue."
            "<|start|>assistant to=user<|message|>"
            '"Guten Abend."'
        )

        visible, reasoning = parse_atem_completion(raw)

        self.assertEqual(visible, '"Guten Abend."')
        self.assertEqual(reasoning, "Reasoning.Continue.")

    def test_visible_atem_marker_remains_invalid(self):
        raw = (
            "Reasoning."
            "<|start|>assistant to=user<|message|>"
            '"Hallo." <|start|>assistant to=self'
        )

        with self.assertRaises(CompletionParseError):
            parse_atem_thinking_prefill(raw)

    def test_direct_visible_text(self):
        visible, reasoning = parse_atem_direct_completion(
            '*Elara schaut auf.* "Rowan?"'
        )
        self.assertEqual(visible, '*Elara schaut auf.* "Rowan?"')
        self.assertEqual(reasoning, "")

    def test_direct_rejects_reasoning(self):
        with self.assertRaises(CompletionParseError):
            parse_atem_direct_completion(
                " to=self<|message|>Internal thoughts"
            )

    def test_strict_reasoning_rejects_double_to(self):
        with self.assertRaisesRegex(
            CompletionParseError, "Unexpected initial"
        ):
            parse_atem_completion(
                ' to to=user<|message|>"Hallo."'
            )

    def test_direct_generation_archives_visible_only(self):
        fmt = self.resolve("muse-atem-direct-v1.json")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = generate_role(
                FakeMuseBackend('"Guten Abend."'),
                [{"role": "user", "content": "Hallo"}],
                {},
                fmt,
                root,
                1,
                "npc",
            )

            self.assertEqual(text, '"Guten Abend."')
            event = next(
                e for e in read_jsonl(root / "events.jsonl")
                if e["type"] == "response"
            )
            self.assertEqual(event["response_status"], "complete")
            self.assertEqual(event["reasoning"], "")
            self.assertEqual(
                event["raw_response"]["choices"][0]["text"],
                '"Guten Abend."',
            )


if __name__ == "__main__":
    unittest.main()
