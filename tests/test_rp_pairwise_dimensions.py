import json

import pytest

from rpbench.rp_pairwise_dimensions import (
    evaluate,
    jobs_for,
    validate_decision,
)


def candidate(name):
    return {
        "id": name,
        "transcript": [
            {"turn": 1, "role": "player", "text": "Hallo."},
            {
                "turn": 1,
                "role": "npc",
                "text": f"{name} legt den Brief ab.",
            },
        ],
    }


def dataset():
    return {
        "schema_version": 1,
        "cases": [candidate(c) for c in "ABCD"],
    }


def verdict():
    return {
        "winner": "X",
        "reason": "NPC X is more consistent.",
        "evidence_x": [
            {"turn": 1, "quote": "A legt den Brief ab."}
        ],
        "evidence_y": [
            {"turn": 1, "quote": "B legt den Brief ab."}
        ],
    }


def test_six_dimensions_balanced():
    jobs = jobs_for(
        dataset(),
        list((
            "german_prose",
            "character_fidelity",
            "dialogue_responsiveness",
            "subtext",
            "narrative_initiative",
            "continuity",
        )),
        1234,
    )
    assert len(jobs) == 72
    assert len(set(jobs)) == 72
    for x, y, criterion in jobs:
        assert (y, x, criterion) in jobs


def test_single_pair_three_dimensions():
    dimensions = [
        "german_prose",
        "character_fidelity",
        "continuity",
    ]
    jobs = jobs_for(dataset(), dimensions, 42, ("A", "C"))
    assert len(jobs) == 6
    assert set((x, y) for x, y, _ in jobs) == {
        ("A", "C"), ("C", "A")
    }


def test_source_validation():
    assert validate_decision(
        verdict(), candidate("A"), candidate("B")
    )["winner"] == "X"


def test_wrong_turn_or_quote_rejected():
    value = verdict()
    value["evidence_y"][0]["turn"] = 2
    with pytest.raises(ValueError, match="Unsupported"):
        validate_decision(value, candidate("A"), candidate("B"))


class FakeBackend:
    model = "test"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.budgets = []

    def prepare(self, messages, sampling, prompt_format):
        self.budgets.append(sampling["max_tokens"])
        return "mock://test", {}

    def generate(self, url, payload):
        finish, content = next(self.replies)
        return content, {
            "choices": [{
                "finish_reason": finish,
                "message": {
                    "content": content,
                    "reasoning": "thinking",
                },
            }],
            "usage": {"completion_tokens": 100},
        }


def test_truncation_retries_at_32k():
    backend = FakeBackend([
        ("length", ""),
        ("stop", json.dumps(verdict())),
    ])
    result = evaluate(
        candidate("A"), candidate("B"),
        "german_prose", "Scene",
        backend,
        {"max_tokens": 16384, "temperature": 1.0},
        32768,
    )

    assert result["valid"] is True
    assert backend.budgets == [16384, 32768]
    assert len(result["attempts"]) == 2


def test_invalid_json_does_not_escalate():
    backend = FakeBackend([("stop", "not json")])
    result = evaluate(
        candidate("A"), candidate("B"),
        "german_prose", "Scene",
        backend,
        {"max_tokens": 16384},
        32768,
    )

    assert not result["valid"]
    assert backend.budgets == [16384]


def test_success_does_not_retry():
    backend = FakeBackend([
        ("stop", json.dumps(verdict()))
    ])
    result = evaluate(
        candidate("A"), candidate("B"),
        "german_prose", "Scene",
        backend,
        {"max_tokens": 16384},
        32768,
    )
    assert result["valid"]
    assert backend.budgets == [16384]


def test_resume_skips_success_and_failures_by_default():
    from rpbench.rp_pairwise_dimensions import plan_jobs

    jobs = [
        ("A", "C", "continuity"),
        ("C", "A", "continuity"),
        ("A", "C", "german_prose"),
    ]
    hashes = {
        "dataset_sha256": "dataset-hash",
        "config_sha256": "config-hash",
        "scenario_sha256": "scenario-hash",
    }
    previous = [
        {
            "x": "A", "y": "C",
            "criterion": "continuity",
            "valid": True,
            "seed": 1234,
            **hashes,
        },
        {
            "x": "C", "y": "A",
            "criterion": "continuity",
            "valid": False,
            "seed": 1234,
            **hashes,
        },
    ]

    assert plan_jobs(jobs, previous, hashes, 1234) == [
        ("A", "C", "german_prose")
    ]

    assert plan_jobs(
        jobs, previous, hashes, 1234, retry_failed=True
    ) == [
        ("C", "A", "continuity"),
        ("A", "C", "german_prose"),
    ]


def test_resume_rejects_changed_dataset():
    from rpbench.rp_pairwise_dimensions import plan_jobs

    job = ("A", "C", "continuity")
    hashes = {
        "dataset_sha256": "new",
        "config_sha256": "same",
        "scenario_sha256": "same",
    }
    previous = [{
        "x": "A",
        "y": "C",
        "criterion": "continuity",
        "seed": 1234,
        "valid": True,
        "dataset_sha256": "old",
        "config_sha256": "same",
        "scenario_sha256": "same",
    }]

    with pytest.raises(ValueError, match="dataset_sha256"):
        plan_jobs([job], previous, hashes, 1234)


def test_resume_rejects_changed_seed():
    from rpbench.rp_pairwise_dimensions import plan_jobs

    job = ("A", "C", "continuity")
    hashes = {
        "dataset_sha256": "d",
        "config_sha256": "c",
        "scenario_sha256": "s",
    }
    previous = [{
        "x": "A", "y": "C",
        "criterion": "continuity",
        "seed": 99,
        "valid": True,
        **hashes,
    }]

    with pytest.raises(ValueError, match="seed"):
        plan_jobs([job], previous, hashes, 1234)


def test_archive_keeps_multiple_attempts(tmp_path):
    from rpbench.rp_pairwise_dimensions import (
        plan_jobs,
        read_archive,
    )

    filename = tmp_path / "results.jsonl"
    hashes = {
        "dataset_sha256": "d",
        "config_sha256": "c",
        "scenario_sha256": "s",
    }
    job = ("A", "C", "continuity")
    base = {
        "x": "A", "y": "C",
        "criterion": "continuity",
        "seed": 1234,
        **hashes,
    }

    with filename.open("a", encoding="utf-8") as file:
        file.write(json.dumps({
            **base, "valid": False
        }) + "\n")
        file.write(json.dumps({
            **base, "valid": True
        }) + "\n")

    archived = read_archive(filename)
    assert len(archived) == 2

    assert plan_jobs(
        [job], archived, hashes, 1234,
        retry_failed=True,
    ) == []


def test_timeout_does_not_retry():
    from rpbench.rp_pairwise_dimensions import evaluate

    class TimeoutBackend:
        model = "timeout-mock"

        def __init__(self):
            self.calls = 0

        def prepare(self, messages, sampling, prompt_format):
            return "mock://timeout", {}

        def generate(self, url, payload):
            self.calls += 1
            raise TimeoutError("Request timed out")

    backend = TimeoutBackend()

    result = evaluate(
        candidate("A"),
        candidate("B"),
        "character_fidelity",
        "Scene",
        backend,
        {"max_tokens": 16384},
        32768,
    )

    assert result["valid"] is False
    assert backend.calls == 1
    assert len(result["attempts"]) == 1
    assert "TimeoutError" in result["attempts"][0]["error"]
