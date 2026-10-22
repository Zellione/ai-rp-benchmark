import tempfile
import unittest
from pathlib import Path

from rpbench.observations import read_jsonl
from rpbench.simulation import generate_role


class FakeNinfer:
    name = "ninfer"
    mode = "chat"

    def prepare(self, messages, sampling, prompt_format):
        return (
            "mock://ninfer",
            {
                "model": "qwen3.8-27b",
                "messages": messages,
                **sampling,
            },
        )

    def generate(self, url, payload):
        return (
            '"Guten Abend."',
            {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": '"Guten Abend."',
                    },
                }],
                "usage": {
                    "completion_tokens": 5,
                },
            },
        )


class ReasoningEffortTests(unittest.TestCase):

    def test_setting_reaches_archived_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            result = generate_role(
                FakeNinfer(),
                [{"role": "user", "content": "Hallo"}],
                {"max_tokens": 16384},
                {},
                root,
                1,
                "player",
                "xhigh",
            )

            self.assertEqual(result, '"Guten Abend."')

            events = read_jsonl(root / "events.jsonl")
            request = next(
                event for event in events
                if event["type"] == "request"
            )

            self.assertEqual(
                request["payload"]["reasoning_effort"],
                "xhigh",
            )

    def test_invalid_effort_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                ValueError, "Invalid reasoning_effort"
            ):
                generate_role(
                    FakeNinfer(), [], {}, {},
                    Path(temp), 1, "player", "extreme",
                )

    def test_none_disables_thinking_in_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            generate_role(
                FakeNinfer(), [], {}, {},
                root, 1, "player", "none",
            )

            request = next(
                event for event in read_jsonl(
                    root / "events.jsonl"
                )
                if event["type"] == "request"
            )

            self.assertEqual(
                request["payload"]["reasoning_effort"],
                "none",
            )


if __name__ == "__main__":
    unittest.main()
