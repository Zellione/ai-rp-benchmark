import tempfile
import unittest
from pathlib import Path

from rpbench.simulation import generate_role
from rpbench.observations import read_jsonl


class FakeBackend:
    name = "llamacpp"
    mode = "chat"

    def __init__(self, contents):
        self.contents = iter(contents)

    def prepare(self, messages, sampling, prompt_format):
        return "mock://generate", {"messages": messages}

    def generate(self, url, payload):
        content = next(self.contents)
        return content, {
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 1},
        }


class OptionalEmptyRetryTests(unittest.TestCase):
    def execute(self, contents, retries):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeBackend(contents)
            error = None

            try:
                answer = generate_role(
                    backend,
                    [{"role": "user", "content": "Continue"}],
                    {},
                    {},
                    root,
                    12,
                    "npc",
                    empty_response_retries=retries,
                )
            except RuntimeError as exc:
                answer = None
                error = str(exc)

            return (
                answer,
                error,
                read_jsonl(root / "events.jsonl"),
            )

    def test_default_does_not_retry_empty(self):
        answer, error, events = self.execute([""], 0)
        self.assertIsNone(answer)
        self.assertIn("empty_response", error)
        self.assertEqual(
            sum(e["type"] == "request" for e in events),
            1,
        )

    def test_opt_in_retry_recovers(self):
        answer, error, events = self.execute(
            ["", '"Come inside."'], 2
        )
        self.assertEqual(answer, '"Come inside."')
        self.assertIsNone(error)
        self.assertEqual(
            sum(e["type"] == "request" for e in events),
            2,
        )

    def test_opt_in_retry_is_bounded(self):
        answer, error, events = self.execute(
            ["", "", ""], 2
        )
        self.assertIsNone(answer)
        self.assertIn("empty_response", error)
        self.assertEqual(
            sum(e["type"] == "request" for e in events),
            3,
        )

    def test_invalid_retry_value(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "0..2"):
                generate_role(
                    FakeBackend(['"Hello"']),
                    [{"role": "user", "content": "Hi"}],
                    {},
                    {},
                    Path(directory),
                    1,
                    "npc",
                    empty_response_retries=True,
                )


if __name__ == "__main__":
    unittest.main()
