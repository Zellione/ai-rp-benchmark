"""Read RPBench conversations without modifying archived runs.

Supports:
- Self-play archives with observations.jsonl
- Fixed-scenario archives with manifest.json + events.jsonl

Both produce normalized player/npc observations.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    result = []

    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number}: invalid JSON"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{number}: expected object"
                )

            result.append(value)

    return result


def validate_observation(entry: dict) -> None:
    if type(entry.get("turn")) is not int or entry["turn"] < 1:
        raise ValueError("Invalid observation turn")

    if entry.get("role") not in {"player", "npc"}:
        raise ValueError("Invalid observation role")

    if not isinstance(entry.get("text"), str):
        raise ValueError("Invalid observation text")


def source_path(run_dir: str | Path) -> Path:
    root = Path(run_dir)

    observations = root / "observations.jsonl"
    if observations.is_file():
        return observations

    events = root / "events.jsonl"
    manifest = root / "manifest.json"

    if events.is_file() and manifest.is_file():
        return events

    raise ValueError(
        f"No supported observation source in {root}"
    )


def validate_sequence(entries: list[dict]) -> None:
    seen = set()
    last_turn = 0
    last_role = None

    for entry in entries:
        validate_observation(entry)

        turn = entry["turn"]
        role = entry["role"]
        key = (turn, role)

        if key in seen:
            raise ValueError(
                f"Duplicate observation: {key}"
            )

        if turn < last_turn:
            raise ValueError("Observations out of order")

        if turn == last_turn:
            if last_role != "player" or role != "npc":
                raise ValueError(
                    "Expected player followed by NPC"
                )
        else:
            if turn != last_turn + 1:
                raise ValueError("Noncontiguous turn sequence")
            if role != "player":
                raise ValueError("Turn must begin with player")
            if last_turn and last_role != "npc":
                raise ValueError(
                    "Previous turn lacks NPC response"
                )

        seen.add(key)
        last_turn = turn
        last_role = role


def read_selfplay(path: Path) -> list[dict]:
    entries = read_jsonl(path)

    normalized = []
    for entry in entries:
        if entry.get("type") != "dialogue_observation":
            raise ValueError(
                "Unexpected self-play observation type"
            )

        validate_observation(entry)

        normalized.append({
            "type": "dialogue_observation",
            "turn": entry["turn"],
            "role": entry["role"],
            "text": entry["text"],
        })

    validate_sequence(normalized)
    return normalized


def read_legacy_selfplay(root: Path) -> list[dict]:
    """Reconstruct early self-play runs from role-tagged responses."""
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    events = read_jsonl(root / "events.jsonl")

    normalized = []

    for event in events:
        if event.get("type") != "response":
            continue

        role = event.get("role")
        if role not in {"player", "npc"}:
            raise ValueError("Invalid legacy self-play response role")

        entry = {
            "type": "dialogue_observation",
            "turn": event.get("turn"),
            "role": role,
            "text": event.get("text"),
        }
        validate_observation(entry)
        normalized.append(entry)

    validate_sequence(normalized)

    # A failed run may contain a completed player response
    # whose corresponding NPC response failed.
    if normalized and normalized[-1]["role"] == "player":
        if manifest.get("status") == "complete":
            raise ValueError(
                "Complete self-play run ends without NPC response"
            )
        normalized.pop()

    if manifest.get("status") == "complete":
        expected = manifest.get("completed_turns")
        if expected is not None and (
            type(expected) is not int
            or len(normalized) != 2 * expected
        ):
            raise ValueError(
                "Completed self-play turn count mismatch"
            )

    validate_sequence(normalized)
    return normalized


def read_fixed_scenario(root: Path) -> list[dict]:
    manifest = json.loads(
        (root / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    scenario = manifest.get("scenario", {})
    turns = scenario.get("turns")

    if not isinstance(turns, list):
        raise ValueError(
            "Legacy manifest has no scenario turns"
        )

    events = read_jsonl(root / "events.jsonl")
    responses = {}

    for event in events:
        if event.get("type") != "response":
            continue

        turn = event.get("turn")
        if type(turn) is not int or not 1 <= turn <= len(turns):
            raise ValueError("Invalid response turn")

        if turn in responses:
            raise ValueError(
                f"Duplicate response for turn {turn}"
            )

        text = event.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"Missing response text at turn {turn}"
            )

        responses[turn] = text

    normalized = []

    for index, user_text in enumerate(turns, 1):
        if index not in responses:
            # Only completed player/NPC pairs are included.
            # A failed run may end with an unanswered user turn.
            break

        if not isinstance(user_text, str):
            raise ValueError(
                f"Invalid scenario turn {index}"
            )

        normalized.extend([
            {
                "type": "dialogue_observation",
                "turn": index,
                "role": "player",
                "text": user_text,
            },
            {
                "type": "dialogue_observation",
                "turn": index,
                "role": "npc",
                "text": responses[index],
            },
        ])

    expected = set(range(1, len(responses) + 1))
    if set(responses) != expected:
        raise ValueError(
            "Legacy responses contain gaps"
        )

    if manifest.get("status") == "complete":
        if len(responses) != len(turns):
            raise ValueError(
                "Complete run has missing responses"
            )

    validate_sequence(normalized)
    return normalized


def read_observations(run_dir: str | Path) -> list[dict]:
    root = Path(run_dir)
    source = source_path(root)

    if source.name == "observations.jsonl":
        return read_selfplay(source)

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("type") == "goal_driven_selfplay":
        return read_legacy_selfplay(root)

    return read_fixed_scenario(root)
