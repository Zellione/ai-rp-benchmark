import json
from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.core import Backend, dump
from rpbench.offline_goal_audit import (
    audit, prepare, validate_verdict,
)


def fixture(tmp_path):
    run = tmp_path / "source"
    run.mkdir()

    definition = {
        "schema_version": 1,
        "goals": {
            "subgoal-01": {
                "criteria": [
                    {
                        "id": "entry",
                        "description": "Inspect visitor register",
                        "evidence_contains": "register",
                    },
                    {
                        "id": "details",
                        "description": (
                            "Obtain a visitor name and arrival time "
                            "from the visitor register."
                        ),
                        "evidence_contains": "time",
                    },
                ],
            },
        },
    }

    manifest = {
        "status": "complete",
        "runtime": {"goal_criteria": definition},
        "completed_turns": 2,
    }
    ground_truth = {
        "schema_version": 1,
        "player": {
            "subgoals": {
                "subgoal-01": {
                    "description": "Inspect visitor register",
                    "initial_status": "pending",
                },
            },
        },
        "characters": {},
    }

    observations = [
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "player",
            "text": "May I see the register?",
        },
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "npc",
            "text": "The key ledger shows 16:20.",
        },
        {
            "type": "dialogue_observation",
            "turn": 2,
            "role": "player",
            "text": "I examine the visitor register.",
        },
        {
            "type": "dialogue_observation",
            "turn": 2,
            "role": "npc",
            "text": (
                "The visitor register shows "
                "Leon Falk arriving at 16:45."
            ),
        },
    ]

    dump(run / "manifest.json", manifest)
    dump(run / "ground_truth.json", ground_truth)

    with (run / "observations.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for item in observations:
            file.write(json.dumps(item) + "\n")

    verdict = {
        "schema_version": 2,
        "goal_id": "subgoal-01",
        "criteria": [
            {
                "criterion_id": "entry",
                "status": "satisfied",
                "first_satisfied_turn": 2,
                "evidence": [{
                    "turn": 2,
                    "role": "player",
                    "quote": "I examine the visitor register.",
                }],
                "reason": "Alex examines the visitor register.",
            },
            {
                "criterion_id": "details",
                "status": "satisfied",
                "first_satisfied_turn": 2,
                "evidence": [{
                    "turn": 2,
                    "role": "npc",
                    "quote": (
                        "The visitor register shows "
                        "Leon Falk arriving at 16:45."
                    ),
                }],
                "reason": (
                    "Leon Falk and 16:45 belong to "
                    "the visitor register, not the key ledger."
                ),
            },
        ],
    }

    return run, definition, observations, verdict


def test_full_transcript_verdict_validates(tmp_path):
    _, definition, observations, verdict = fixture(tmp_path)

    result = validate_verdict(
        verdict, "subgoal-01", definition, observations
    )

    assert result["criteria"][1]["first_satisfied_turn"] == 2


def test_fabricated_document_quote_is_rejected(tmp_path):
    _, definition, observations, verdict = fixture(tmp_path)

    corrupted = deepcopy(verdict)
    corrupted["criteria"][1]["evidence"][0]["quote"] = (
        "The key ledger shows Leon Falk arriving at 16:20."
    )

    with pytest.raises(ValueError, match="source mismatch"):
        validate_verdict(
            corrupted, "subgoal-01", definition, observations
        )


def test_premature_completion_is_rejected(tmp_path):
    _, definition, observations, verdict = fixture(tmp_path)

    corrupted = deepcopy(verdict)
    corrupted["criteria"][1]["first_satisfied_turn"] = 1

    with pytest.raises(ValueError, match="precedes"):
        validate_verdict(
            corrupted, "subgoal-01", definition, observations
        )


def test_audit_is_idempotent_and_preserves_source(
    tmp_path, monkeypatch
):
    run, _, _, verdict = fixture(tmp_path)
    output = tmp_path / "independent-audit"

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")

    calls = []

    def fake_generate(self, url, payload):
        calls.append((url, payload))
        return json.dumps(verdict), {
            "choices": [{
                "message": {"content": json.dumps(verdict)},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 100},
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    before = {
        filename.name: filename.read_bytes()
        for filename in run.iterdir()
    }

    options = dict(
        run=run,
        output=output,
        goal_id="subgoal-01",
        model="test-glm-judge",
    )

    first = audit(**options)
    second = audit(**options)

    assert first == second
    assert len(calls) == 1
    assert calls[0][0].endswith(
        "/api/subscription/v1/chat/completions"
    )

    after = {
        filename.name: filename.read_bytes()
        for filename in run.iterdir()
    }
    assert before == after


def test_audit_identity_rejects_changed_source(
    tmp_path, monkeypatch
):
    run, _, _, verdict = fixture(tmp_path)
    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")

    def fake_generate(self, url, payload):
        return json.dumps(verdict), {
            "choices": [{
                "message": {"content": json.dumps(verdict)},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(Backend, "generate", fake_generate)

    output = tmp_path / "audit"
    audit(run, output, "subgoal-01", "fixture-model")

    with (run / "observations.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write("\n")

    with pytest.raises(ValueError, match="identity mismatch"):
        audit(run, output, "subgoal-01", "fixture-model")



def test_invalid_raw_response_is_reusable(
    tmp_path, monkeypatch
):
    run, _, _, verdict = fixture(tmp_path)
    invalid = deepcopy(verdict)
    invalid["criteria"][1]["first_satisfied_turn"] = 1

    calls = []

    def fake_generate(self, url, payload):
        calls.append(url)
        return json.dumps(invalid), {
            "choices": [{"finish_reason": "stop"}],
        }

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")
    monkeypatch.setattr(Backend, "generate", fake_generate)

    output = tmp_path / "invalid-audit"

    options = dict(
        run=run,
        output=output,
        goal_id="subgoal-01",
        model="fixture-model",
    )

    with pytest.raises(ValueError, match="precedes"):
        audit(**options)

    assert (output / "identity.json").exists()
    assert (output / "raw_response.json").exists()
    assert not (output / "verdict.json").exists()

    monkeypatch.delenv("NANOGPT_API_KEY")

    with pytest.raises(ValueError, match="precedes"):
        audit(**options)

    assert len(calls) == 1


def test_raw_response_rejects_changed_source(
    tmp_path, monkeypatch
):
    run, _, _, verdict = fixture(tmp_path)
    invalid = deepcopy(verdict)
    invalid["criteria"][1]["first_satisfied_turn"] = 1

    def fake_generate(self, url, payload):
        return json.dumps(invalid), {
            "choices": [{"finish_reason": "stop"}],
        }

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")
    monkeypatch.setattr(Backend, "generate", fake_generate)

    output = tmp_path / "invalid-audit"

    with pytest.raises(ValueError, match="precedes"):
        audit(run, output, "subgoal-01", "fixture-model")

    with (run / "observations.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write("\n")

    with pytest.raises(ValueError, match="identity mismatch"):
        audit(run, output, "subgoal-01", "fixture-model")



def external_configuration(tmp_path, definition):
    path = tmp_path / "quest-v4.json"
    dump(path, {
        "runtime": {
            "goal_criteria": definition,
        },
    })
    return path


def test_external_goal_definition_changes_identity(tmp_path):
    run, definition, _, _ = fixture(tmp_path)

    revised = deepcopy(definition)
    revised["goals"]["subgoal-01"]["criteria"][1][
        "description"
    ] = "Precisely identify visitor and timestamp."

    config = external_configuration(tmp_path, revised)

    original, original_definition, _, old_context = prepare(
        run, "subgoal-01", "fixture-model",
        "https://nano-gpt.com/api/subscription", 16384,
    )

    changed, changed_definition, _, new_context = prepare(
        run, "subgoal-01", "fixture-model",
        "https://nano-gpt.com/api/subscription", 16384,
        goal_definition=config,
    )

    assert original["source_files_sha256"] == (
        changed["source_files_sha256"]
    )
    assert original["definition_sha256"] != (
        changed["definition_sha256"]
    )
    assert old_context["criteria"] != new_context["criteria"]
    assert changed_definition == revised
    assert original_definition == definition


def test_external_goal_definition_rejects_structure_change(tmp_path):
    run, definition, _, _ = fixture(tmp_path)
    revised = deepcopy(definition)
    revised["goals"]["subgoal-01"]["criteria"][1][
        "id"
    ] = "different-criterion"

    config = external_configuration(tmp_path, revised)

    with pytest.raises(ValueError, match="criterion ID mismatch"):
        prepare(
            run, "subgoal-01", "fixture-model",
            "https://nano-gpt.com/api/subscription", 16384,
            goal_definition=config,
        )


def test_external_goal_definition_rejects_missing_file(tmp_path):
    run, _, _, _ = fixture(tmp_path)

    with pytest.raises(FileNotFoundError):
        prepare(
            run, "subgoal-01", "fixture-model",
            "https://nano-gpt.com/api/subscription", 16384,
            goal_definition=tmp_path / "missing.json",
        )


def test_external_definition_prevents_verdict_reuse(
    tmp_path, monkeypatch
):
    run, definition, _, verdict = fixture(tmp_path)

    revised = deepcopy(definition)
    revised["goals"]["subgoal-01"]["criteria"][0][
        "description"
    ] = "Revised visitor register requirement."

    config = external_configuration(tmp_path, revised)

    calls = []

    def fake_generate(self, url, payload):
        calls.append(url)
        return json.dumps(verdict), {
            "choices": [{"finish_reason": "stop"}],
        }

    monkeypatch.setenv("NANOGPT_API_KEY", "fixture-key")
    monkeypatch.setattr(Backend, "generate", fake_generate)

    output = tmp_path / "audit"

    audit(
        run, output, "subgoal-01", "fixture-model"
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        audit(
            run, output, "subgoal-01", "fixture-model",
            goal_definition=config,
        )

    assert len(calls) == 1
