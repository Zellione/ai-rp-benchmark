"""Resume archived goal-driven self-play without replaying completed work."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .core import append, dump, load
from .run_lock import run_lock
from .observations import read_jsonl, validate_sequence
from .simulation import (
    generate_role,
    make_backend,
    make_histories,
)
from .state import make_observation


def _now():
    return datetime.now(timezone.utc).isoformat()


def _entries(root):
    path = root / "observations.jsonl"
    entries = read_jsonl(path) if path.exists() else []

    for entry in entries:
        if entry.get("type") != "dialogue_observation":
            raise ValueError("Unexpected observation type")

    validate_sequence(entries)
    return entries


def _recover_archived_responses(root, entries, planned):
    """Recover successful API responses written before their observations."""
    path = root / "events.jsonl"
    events = read_jsonl(path) if path.exists() else []

    complete = {}
    for event in events:
        if event.get("type") != "response":
            continue
        if event.get("response_status") != "complete":
            continue

        turn = event.get("turn")
        role = event.get("role")
        text = event.get("text")

        if (
            type(turn) is not int
            or role not in {"player", "npc"}
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Invalid completed response event")

        key = (turn, role)
        if key in complete and complete[key] != text:
            raise ValueError(
                f"Conflicting completed responses: {key}"
            )
        complete[key] = text

    observed = {
        (entry["turn"], entry["role"]): entry["text"]
        for entry in entries
    }

    for key, text in observed.items():
        if key in complete and complete[key] != text:
            raise ValueError(
                f"Observation differs from archived response: {key}"
            )

    while True:
        index = len(entries)
        turn = index // 2 + 1
        role = "player" if index % 2 == 0 else "npc"

        if turn > planned:
            break

        key = (turn, role)
        if key not in complete:
            break

        entry = make_observation(turn, role, complete[key])
        append(root / "observations.jsonl", entry)
        entries.append(entry)

    validate_sequence(entries)

    recovered_keys = {
        (entry["turn"], entry["role"])
        for entry in entries
    }

    if not set(complete).issubset(recovered_keys):
        raise ValueError(
            "Completed response events contain gaps or future turns"
        )

    return entries


def _render_pair(turn, player_name, player_text, npc_text):
    return (
        f"## Turn {turn}\n\n"
        f"### Player ({player_name})\n\n"
        f"{player_text}\n\n"
        f"### NPC\n\n{npc_text}\n\n"
    )


def _sync_transcript(root, manifest, entries):
    """Append missing complete turns; never overwrite transcript bytes."""
    path = root / "conversation.md"

    if not path.is_file():
        raise ValueError("Missing conversation.md")

    complete_turns = len(entries) // 2
    expected = (
        f"# {manifest['scenario'].get('title', manifest['scenario'].get('id'))}"
        "\n\n"
        f"Player: {manifest['player']['name']}\n\n"
        "Mode: Goal-driven self-play\n\n"
    )

    for index in range(complete_turns):
        player = entries[index * 2]
        npc = entries[index * 2 + 1]
        turn = index + 1

        if (
            player["turn"] != turn
            or player["role"] != "player"
            or npc["turn"] != turn
            or npc["role"] != "npc"
        ):
            raise ValueError("Invalid player/NPC turn order")

        expected += _render_pair(
            turn,
            manifest["player"]["name"],
            player["text"],
            npc["text"],
        )

    existing = path.read_text(encoding="utf-8")
    if not expected.startswith(existing):
        raise ValueError(
            "Transcript differs from archived observations; "
            "refusing to overwrite"
        )

    missing = expected[len(existing):]
    if missing:
        with path.open("a", encoding="utf-8") as f:
            f.write(missing)
            f.flush()
            os.fsync(f.fileno())

    return complete_turns


def _histories(manifest, entries):
    opening = manifest["scenario"].get(
        "opening",
        "Begin the roleplay. Introduce your character "
        "and take one natural action.",
    )

    player_messages, npc_messages = make_histories(
        manifest["rendered_player_system_prompt"],
        manifest["rendered_npc_system_prompt"],
        opening,
    )

    for entry in entries:
        role = entry["role"]
        text = entry["text"]

        if role == "player":
            npc_messages.append({
                "role": "user",
                "content": text,
            })
        else:
            player_text = entries[
                2 * (entry["turn"] - 1)
            ]["text"]

            player_messages.extend([
                {"role": "assistant", "content": player_text},
                {"role": "user", "content": text},
            ])

            npc_messages.append({
                "role": "assistant",
                "content": text,
            })

    return player_messages, npc_messages


def resume_simulation(run_dir, turns=None):
    root = Path(run_dir).resolve()
    with run_lock(root):
        return _resume_simulation_locked(root, turns)


def _resume_simulation_locked(run_dir, turns=None):
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = load(manifest_path)

    if manifest.get("type") != "goal_driven_selfplay":
        raise ValueError("Not a self-play run")

    old_limit = manifest["planned_turns"]
    target = old_limit if turns is None else turns

    if (
        type(target) is not int
        or not 1 <= target <= 400
        or target < old_limit
    ):
        raise ValueError(
            "Target turns must be between the original "
            "limit and 400"
        )

    entries = _entries(root)

    if len(entries) > target * 2:
        raise ValueError("More observations than requested turns")

    # Validate complete-response provenance before changing files.
    events_path = root / "events.jsonl"
    events = (
        read_jsonl(events_path)
        if events_path.exists()
        else []
    )
    completed_events = {
        (event.get("turn"), event.get("role")):
            event.get("text")
        for event in events
        if event.get("type") == "response"
        and event.get("response_status") == "complete"
    }
    for entry in entries:
        key = (entry["turn"], entry["role"])
        if key in completed_events and (
            completed_events[key] != entry["text"]
        ):
            raise ValueError(
                f"Archived response mismatch: {key}"
            )

    backend_config = {
        "backend": manifest["backend"],
    }
    backend = make_backend(backend_config)

    if backend.mode != "chat":
        raise ValueError("Resume requires chat mode")

    # Recover journaled successful responses first.
    entries = _recover_archived_responses(
        root, entries, target
    )

    completed = _sync_transcript(
        root, manifest, entries
    )

    manifest["completed_turns"] = completed
    manifest["planned_turns"] = target
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest.pop("finished_utc", None)

    append(root / "events.jsonl", {
        "type": "resume",
        "timestamp": _now(),
        "completed_turns": completed,
        "next_role": (
            "npc" if len(entries) % 2 else "player"
        ),
        "target_turns": target,
    })
    dump(manifest_path, manifest)

    player_messages, npc_messages = _histories(
        manifest, entries
    )

    sampling = manifest["sampling"]
    player_sampling = manifest["player_sampling"]
    prompt_format = manifest["prompt_format"]
    player_format = manifest["player_prompt_format"]

    try:
        for turn in range(completed + 1, target + 1):
            have_player = (
                len(entries) % 2 == 1
                and entries[-1]["turn"] == turn
                and entries[-1]["role"] == "player"
            )

            if have_player:
                player_text = entries[-1]["text"]
            else:
                player_text = generate_role(
                    backend,
                    player_messages,
                    player_sampling,
                    player_format,
                    root,
                    turn,
                    "player",
                )

                entry = make_observation(
                    turn, "player", player_text
                )
                append(root / "observations.jsonl", entry)
                entries.append(entry)
                npc_messages.append({
                    "role": "user",
                    "content": player_text,
                })

            npc_text = generate_role(
                backend,
                npc_messages,
                sampling,
                prompt_format,
                root,
                turn,
                "npc",
            )

            entry = make_observation(
                turn, "npc", npc_text
            )
            append(root / "observations.jsonl", entry)
            entries.append(entry)

            player_messages.extend([
                {"role": "assistant", "content": player_text},
                {"role": "user", "content": npc_text},
            ])
            npc_messages.append({
                "role": "assistant",
                "content": npc_text,
            })

            completed = _sync_transcript(
                root, manifest, entries
            )

            manifest["completed_turns"] = completed
            dump(manifest_path, manifest)

        manifest["status"] = "complete"

    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise

    finally:
        manifest["finished_utc"] = _now()
        dump(manifest_path, manifest)

    return root


def main():
    parser = argparse.ArgumentParser(
        description="Resume an archived RPBench self-play run"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--turns", type=int)
    args = parser.parse_args()

    print(resume_simulation(args.run, args.turns))


if __name__ == "__main__":
    main()
