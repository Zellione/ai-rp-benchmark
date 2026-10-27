"""Knowledge gate integration, provenance, and resume behavior."""

import json

import pytest

import rpbench.evaluator_runtime as runtime
from rpbench.core import load
from rpbench.knowledge_gate_journal import read_gate_events
from rpbench.observations import read_jsonl


PLAYER = {"known_facts": []}
STATE = {"goals": {}}

SETTINGS = {
    "mode": "llm",
    "id": "gate-runtime-test",
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:18081",
        "model": "mock-gate-evaluator",
    },
    "sampling": {"max_tokens": 512},
    "max_attempts": 1,
}


def observations(npc):
    return [
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "player",
            "text": "Elena leaves the archive.",
        },
        {
            "type": "dialogue_observation",
            "turn": 1,
            "role": "npc",
            "text": npc,
        },
    ]


def reconcile(root, entries):
    return runtime.reconcile_evaluations(
        root,
        PLAYER,
        entries,
        STATE,
        [],
        SETTINGS,
        knowledge_only=True,
    )


def test_offscreen_scene_skips_knowledge_llm(
    tmp_path, monkeypatch,
):
    npc = "Mara and Voss discuss a secret after Elena leaves."
    entries = observations(npc)
    calls = {"gate": 0, "knowledge": 0}

    def gate(player_action, npc_text, settings, on_response=None):
        calls["gate"] += 1
        assert player_action == "Elena leaves the archive."
        assert npc_text == npc
        return ["offscreen"], {"usage": None}

    def knowledge(*args, **kwargs):
        calls["knowledge"] += 1
        raise AssertionError("Knowledge LLM must not be called")

    monkeypatch.setattr(runtime, "evaluate_gate", gate)
    monkeypatch.setattr(runtime, "evaluate_llm", knowledge)

    result = reconcile(tmp_path, entries)

    assert calls == {"gate": 1, "knowledge": 0}
    assert result["facts"] == []

    events = read_jsonl(tmp_path / "events.jsonl")
    assert len([
        e for e in events
        if e.get("type") == "knowledge_gate_v1"
    ]) == 1

    evaluation = next(
        e for e in events
        if e.get("type") == "player_evaluation_v1"
    )
    assert evaluation["result"]["knowledge_updates"] == []

    reconcile(tmp_path, entries)
    assert calls == {"gate": 1, "knowledge": 0}


def test_resume_reuses_gate_after_knowledge_failure(
    tmp_path, monkeypatch,
):
    npc = (
        "Mara tells Elena the reading room is open.\n\n"
        "Later, Voss privately reveals a vault secret."
    )
    entries = observations(npc)
    calls = {"gate": 0, "knowledge": 0}
    fail = {"enabled": True}

    def gate(player_action, npc_text, settings, on_response=None):
        calls["gate"] += 1
        return (
            ["observable", "offscreen"],
            {"usage": {"completion_tokens": 10}},
        )

    def knowledge(
        npc_text, state, known, settings,
        on_response=None, knowledge_only=False, repair=None,
    ):
        calls["knowledge"] += 1

        assert npc_text == (
            "Mara tells Elena the reading room is open."
        )
        assert "vault secret" not in npc_text
        assert settings["_player_action"] == (
            "Elena leaves the archive."
        )
        assert knowledge_only is True

        if fail["enabled"]:
            raise ValueError("Simulated extraction failure")

        return {
            "knowledge_updates": [{
                "fact": "Mara says the reading room is open.",
                "source_turns": [1],
            }],
            "goal_updates": [],
        }

    monkeypatch.setattr(runtime, "evaluate_gate", gate)
    monkeypatch.setattr(runtime, "evaluate_llm", knowledge)

    with pytest.raises(ValueError, match="Simulated"):
        reconcile(tmp_path, entries)

    archived = read_gate_events(
        tmp_path / "events.jsonl", entries
    )
    assert archived[1]["labels"] == [
        "observable", "offscreen"
    ]

    fail["enabled"] = False
    result = reconcile(tmp_path, entries)

    assert calls == {"gate": 1, "knowledge": 2}
    assert any(
        fact["fact"] == "Mara says the reading room is open."
        for fact in result["facts"]
    )

    reconcile(tmp_path, entries)
    assert calls == {"gate": 1, "knowledge": 2}


def test_tampered_gate_event_rejected(
    tmp_path, monkeypatch,
):
    npc = "Mara privately speaks to Voss."
    entries = observations(npc)

    def gate(player_action, npc_text, settings, on_response=None):
        return ["offscreen"], {"usage": None}

    monkeypatch.setattr(runtime, "evaluate_gate", gate)

    reconcile(tmp_path, entries)

    path = tmp_path / "events.jsonl"
    events = read_jsonl(path)

    for event in events:
        if event.get("type") == "knowledge_gate_v1":
            event["npc_sha256"] = "0" * 64

    path.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        read_gate_events(path, entries)

    with pytest.raises(ValueError, match="provenance mismatch"):
        reconcile(tmp_path, entries)


def test_gate_source_change_rejected(
    tmp_path, monkeypatch,
):
    entries = observations("Mara tells Elena hello.")

    monkeypatch.setattr(
        runtime,
        "evaluate_gate",
        lambda *args, **kwargs: (
            ["observable"], {"usage": None}
        ),
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_llm",
        lambda *args, **kwargs: {
            "knowledge_updates": [],
            "goal_updates": [],
        },
    )

    reconcile(tmp_path, entries)

    changed = observations("Mara tells Elena goodbye.")

    with pytest.raises(ValueError, match="provenance mismatch"):
        read_gate_events(
            tmp_path / "events.jsonl",
            changed,
        )
