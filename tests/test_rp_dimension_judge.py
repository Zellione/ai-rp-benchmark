import json

import pytest

from rpbench.rp_dimension_judge import (
    evaluate,
    pending_jobs,
    plan_jobs,
    validate_verdict,
)


def candidate(name="A"):
    return {
        "id": name,
        "transcript": [
            {"turn": 1, "role": "player", "text": "Hallo."},
            {
                "turn": 1,
                "role": "npc",
                "text": "Elisa legt die Pinzette ab.",
            },
        ],
    }


def dataset():
    return {
        "schema_version": 1,
        "cases": [candidate("A"), candidate("B")],
    }


def verdict():
    return {
        "score": 7,
        "reason": "Consistent, restrained German.",
        "evidence": [
            {
                "turn": 1,
                "quote": "Elisa legt die Pinzette ab.",
            }
        ],
    }


def test_valid_verdict():
    assert validate_verdict(
        verdict(), candidate()["transcript"]
    )["score"] == 7


def test_wrong_turn_rejected():
    value = verdict()
    value["evidence"][0]["turn"] = 2

    with pytest.raises(ValueError, match="Unsupported"):
        validate_verdict(value, candidate()["transcript"])


def test_player_quote_rejected():
    value = verdict()
    value["evidence"][0]["quote"] = "Hallo."

    with pytest.raises(ValueError, match="Unsupported"):
        validate_verdict(value, candidate()["transcript"])


def test_jobs_are_reproducible():
    dimensions = ["german_prose", "continuity"]
    first = plan_jobs(dataset(), dimensions, None, 1234)

    assert len(first) == 4
    assert first == plan_jobs(dataset(), dimensions, None, 1234)


def test_resume_preserves_successes_and_failures():
    jobs = [
        ("A", "german_prose"),
        ("A", "continuity"),
        ("B", "german_prose"),
    ]
    hashes = {"dataset_sha256": "d", "config_sha256": "c"}
    records = [
        {
            "candidate": "A", "dimension": "german_prose",
            "valid": True, **hashes,
        },
        {
            "candidate": "A", "dimension": "continuity",
            "valid": False, **hashes,
        },
    ]

    assert pending_jobs(jobs, records, hashes) == [
        ("B", "german_prose")
    ]
    assert pending_jobs(
        jobs, records, hashes, retry_failed=True
    ) == [
        ("A", "continuity"),
        ("B", "german_prose"),
    ]


def test_resume_rejects_changed_config():
    hashes = {"config_sha256": "new"}

    with pytest.raises(ValueError, match="config_sha256"):
        pending_jobs(
            [("A", "continuity")],
            [{
                "candidate": "A", "dimension": "continuity",
                "valid": True, "config_sha256": "old",
            }],
            hashes,
        )


class FakeBackend:
    model = "fake"

    def __init__(self, finish="stop"):
        self.finish = finish
        self.calls = 0

    def prepare(self, messages, sampling, fmt):
        return "mock://judge", {}

    def generate(self, url, payload):
        self.calls += 1
        content = json.dumps(verdict())
        return content, {
            "choices": [{
                "message": {
                    "content": content,
                    "reasoning_content": "thinking",
                },
                "finish_reason": self.finish,
            }],
            "usage": {"completion_tokens": 120},
        }


def test_evaluate_success_single_call():
    backend = FakeBackend()
    result = evaluate(
        candidate(), "german_prose", "Scene",
        backend, {"max_tokens": 16384},
    )
    assert result["valid"]
    assert result["verdict"]["score"] == 7
    assert backend.calls == 1


def test_truncation_is_recorded_without_retry():
    backend = FakeBackend(finish="length")
    result = evaluate(
        candidate(), "continuity", "Scene",
        backend, {"max_tokens": 16384},
    )
    assert not result["valid"]
    assert result["finish_reason"] == "length"
    assert backend.calls == 1


def test_timeout_is_recorded_without_retry():
    backend = FakeBackend()

    def fail(url, payload):
        backend.calls += 1
        raise TimeoutError("Read timed out")

    backend.generate = fail
    result = evaluate(
        candidate(), "subtext", "Scene",
        backend, {"max_tokens": 16384},
    )
    assert not result["valid"]
    assert "TimeoutError" in result["error"]
    assert backend.calls == 1
