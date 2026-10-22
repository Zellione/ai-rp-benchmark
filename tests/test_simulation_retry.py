import json
import tempfile
import unittest
from pathlib import Path

from rpbench.simulation import generate_role


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def prepare(self, messages, sampling, prompt_format):
        return "mock://generate", {"messages": messages}

    def generate(self, url, payload):
        self.calls += 1
        return next(self.responses)


def response(text, reasoning="", finish="stop"):
    return (
        text,
        {
            "choices": [{
                "finish_reason": finish,
                "message": {
                    "content": text,
                    "reasoning_content": reasoning,
                },
            }],
            "usage": {
                "completion_tokens": 20,
            },
        },
    )


class SimulationRetryTests(unittest.TestCase):

    def execute(self, responses):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = FakeBackend(responses)

            try:
                answer = generate_role(
                    backend,
                    [{"role": "user", "content": "Hello"}],
                    {},
                    {},
                    root,
                    5,
                    "player",
                )
                error = None
            except RuntimeError as exc:
                answer = None
                error = str(exc)

            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            return answer, error, backend.calls, events

    def test_reasoning_only_retried(self):
        answer, error, calls, events = self.execute([
            response("", "Drafted an answer."),
            response('"Hello."', "Considered the scene."),
        ])

        self.assertEqual(answer, '"Hello."')
        self.assertIsNone(error)
        self.assertEqual(calls, 2)
        self.assertEqual(
            sum(e["type"] == "generation_retry" for e in events),
            1,
        )
        self.assertEqual(
            [
                e["response_status"]
                for e in events if e["type"] == "response"
            ],
            ["reasoning_only", "complete"],
        )

    def test_retry_only_once(self):
        answer, error, calls, events = self.execute([
            response("", "First reasoning."),
            response("", "Second reasoning."),
        ])
        self.assertIsNone(answer)
        self.assertIn("reasoning_only", error)
        self.assertEqual(calls, 2)

    def test_no_retry_for_truncation(self):
        answer, error, calls, events = self.execute([
            response("Partial text", finish="length"),
        ])
        self.assertIsNone(answer)
        self.assertIn("truncated", error)
        self.assertEqual(calls, 1)

    def test_no_retry_for_empty_without_reasoning(self):
        answer, error, calls, events = self.execute([
            response(""),
        ])
        self.assertIsNone(answer)
        self.assertIn("empty_response", error)
        self.assertEqual(calls, 1)

    def test_normal_response_uses_one_request(self):
        answer, error, calls, events = self.execute([
            response("*Rowan nods.*"),
        ])
        self.assertEqual(answer, "*Rowan nods.*")
        self.assertIsNone(error)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
