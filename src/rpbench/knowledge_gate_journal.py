"""Durable, source-bound knowledge-gate decisions."""

from hashlib import sha256
from pathlib import Path

from .core import append
from .knowledge_gate import split_paragraphs, validate_gate_result
from .observations import read_jsonl


EVENT_TYPE = "knowledge_gate_v1"


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def make_gate_event(turn, player_text, npc_text, labels):
    if type(turn) is not int or turn < 1:
        raise ValueError("Invalid gate turn")

    if not isinstance(player_text, str):
        raise ValueError("Invalid player text")

    if not isinstance(npc_text, str):
        raise ValueError("Invalid NPC text")

    parts = split_paragraphs(npc_text)

    validated = validate_gate_result({
        "labels": [
            {"id": index, "visibility": label}
            for index, label in enumerate(labels)
        ],
    }, len(parts))

    return {
        "type": EVENT_TYPE,
        "turn": turn,
        "player_sha256": _hash(player_text),
        "npc_sha256": _hash(npc_text),
        "labels": validated,
    }


def read_gate_events(events_path, observations):
    """Validate archived decisions against original observations."""
    path = Path(events_path)

    events = read_jsonl(path) if path.exists() else []

    observed = {
        (item["turn"], item["role"]): item["text"]
        for item in observations
    }

    decisions = {}

    for event in events:
        if event.get("type") != EVENT_TYPE:
            continue

        if set(event) != {
            "type", "turn", "player_sha256",
            "npc_sha256", "labels",
        }:
            raise ValueError("Invalid archived gate fields")

        turn = event["turn"]

        if type(turn) is not int or turn < 1:
            raise ValueError("Invalid archived gate turn")

        if turn in decisions:
            raise ValueError("Duplicate gate decision")

        player = observed.get((turn, "player"))
        npc = observed.get((turn, "npc"))

        if player is None or npc is None:
            raise ValueError("Gate observation missing")

        expected = make_gate_event(
            turn, player, npc, event["labels"]
        )

        if event != expected:
            raise ValueError("Gate provenance mismatch")

        decisions[turn] = event

    return decisions


def append_gate_event(
    events_path, observations,
    turn, player_text, npc_text, labels,
):
    event = make_gate_event(
        turn, player_text, npc_text, labels
    )

    existing = read_gate_events(events_path, observations)

    if turn in existing:
        raise ValueError("Gate decision already archived")

    append(Path(events_path), event)
    return event
