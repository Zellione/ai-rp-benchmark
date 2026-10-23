import tempfile
import unittest
from pathlib import Path

from rpbench.completion_response import (
    CompletionParseError,
    parse_atem_completion,
)
from rpbench.observations import read_jsonl
from rpbench.simulation import generate_role


class FakeATEMBackend:
    name = "llamacpp"
    mode = "completion"

    def __init__(self, raw_text):
        self.raw_text = raw_text

    def prepare(self, messages, sampling, prompt_format):
        return "mock://atem", {"prompt": "test"}

    def generate(self, url, payload):
        return self.raw_text, {
            "choices": [{
                "finish_reason": "stop",
                "text": self.raw_text,
            }],
            "usage": {"completion_tokens": 50},
        }


class ATEMCompletionTests(unittest.TestCase):

    def test_reasoning_and_visible_response(self):
        raw = (
            " to=self<|message|>Think about Elara."
            "<|start|>assistant to=user<|message|>"
            '*Elara looks up.* "Rowan?"'
        )

        visible, reasoning = parse_atem_completion(raw)

        self.assertEqual(
            visible,
            '*Elara looks up.* "Rowan?"',
        )
        self.assertEqual(reasoning, "Think about Elara.")

    def test_direct_visible_response(self):
        visible, reasoning = parse_atem_completion(
            ' to=user<|message|>"Hallo."<|eot|>'
        )
        self.assertEqual(visible, '"Hallo."')
        self.assertEqual(reasoning, "")

    def test_unexpected_channel_rejected(self):
        with self.assertRaises(CompletionParseError):
            parse_atem_completion(
                " to=other<|message|>Unexpected"
            )

    def test_no_visible_channel_rejected(self):
        with self.assertRaises(CompletionParseError):
            parse_atem_completion(
                " to=self<|message|>Only reasoning."
            )

    def test_generation_archives_both_channels(self):
        raw = (
            " to=self<|message|>Internal reasoning."
            "<|start|>assistant to=user<|message|>"
            '*Elara sieht auf.* "Rowan?"'
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            visible = generate_role(
                FakeATEMBackend(raw),
                [{"role": "user", "content": "Hallo"}],
                {},
                {"response_parser": "atem_v1"},
                root,
                1,
                "npc",
            )

            self.assertEqual(
                visible,
                '*Elara sieht auf.* "Rowan?"',
            )

            events = read_jsonl(root / "events.jsonl")
            response = next(
                e for e in events
                if e["type"] == "response"
            )

            self.assertEqual(
                response["response_status"],
                "complete",
            )
            self.assertEqual(
                response["reasoning"],
                "Internal reasoning.",
            )
            self.assertEqual(
                response["raw_response"]["choices"][0]["text"],
                raw,
            )

    def test_parse_error_preserves_raw_response(self):
        raw = " to=self<|message|>No final response"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            with self.assertRaisesRegex(
                RuntimeError, "parse_error"
            ):
                generate_role(
                    FakeATEMBackend(raw),
                    [],
                    {},
                    {"response_parser": "atem_v1"},
                    root,
                    1,
                    "npc",
                )

            events = read_jsonl(root / "events.jsonl")
            response = next(
                e for e in events
                if e["type"] == "response"
            )

            self.assertEqual(
                response["response_status"],
                "parse_error",
            )
            self.assertEqual(
                response["raw_response"]["choices"][0]["text"],
                raw,
            )


if __name__ == "__main__":
    unittest.main()
