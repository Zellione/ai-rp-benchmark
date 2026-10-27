"""Resume archived goal-driven self-play without replaying completed work."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .core import append, dump, load
from .progress_status import emit
from .run_lock import run_lock
from .journal_recovery import recover_run_journals
from .observations import read_jsonl, validate_sequence
from .simulation import (
    generate_role,
    make_backend,
    make_histories,
)
from .state import make_observation, initial_state
from .player_controller import render_controlled_player_request
from .goal_progress import refresh_goal_progress
from .evaluator_runtime import (
    validate_evaluator_config,
    reconcile_evaluations,
)
from .player_knowledge import (
    replay_player_knowledge,
    update_player_knowledge,
    validate_knowledge_rules,
)
from .simulation_st_config import render_st_npc_request


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

    npc_opening = manifest["scenario"].get("npc_opening")
    if npc_opening is not None:
        if not isinstance(npc_opening, str) or not npc_opening.strip():
            raise ValueError("Invalid archived NPC opening")
        expected += f"## NPC Opening\n\n{npc_opening}\n\n"

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
        manifest["scenario"].get("npc_opening"),
        strict_alternation=manifest.get(
            "runtime", {}
        ).get("npc_opening_strict_alternation", False),
        user_filler=manifest.get("runtime", {})
        .get("sillytavern_chat_v1", {})
        .get("user_filler"),
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


def resume_simulation(
    run_dir, turns=None,
    evaluator_max_tokens=None,
    evaluator_attempts=None,
    evaluator_timeout=None,
    evaluator_temperature=None,
    evaluator_top_p=None,
    evaluator_top_k=None,
):
    root = Path(run_dir).resolve()
    with run_lock(root):
        return _resume_simulation_locked(
            root, turns,
            evaluator_max_tokens=evaluator_max_tokens,
            evaluator_attempts=evaluator_attempts,
            evaluator_timeout=evaluator_timeout,
            evaluator_temperature=evaluator_temperature,
            evaluator_top_p=evaluator_top_p,
            evaluator_top_k=evaluator_top_k,
        )


def _resume_simulation_locked(
    run_dir, turns=None,
    evaluator_max_tokens=None,
    evaluator_attempts=None,
    evaluator_timeout=None,
    evaluator_temperature=None,
    evaluator_top_p=None,
    evaluator_top_k=None,
):
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = load(manifest_path)

    if manifest.get("type") != "goal_driven_selfplay":
        raise ValueError("Not a self-play run")

    # Safe to inspect and repair journals: this process owns the lock.
    recover_run_journals(root)

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

    if backend.mode not in {"chat", "completion"}:
        raise ValueError(
            "Resume requires chat or completion mode"
        )

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

    npc_renderer = manifest.get("npc_renderer")
    if npc_renderer is not None and backend.mode != "chat":
        raise ValueError(
            "Archived st_chat_npc requires a chat backend"
        )

    controlled_player = manifest.get(
        "runtime", {}
    ).get("controlled_player", False)

    if type(controlled_player) is not bool:
        raise ValueError("Invalid archived controlled_player option")

    player_state = (
        load(root / "state.json")
        if controlled_player else None
    )

    if controlled_player:
        knowledge_rules = manifest.get(
            "runtime", {}
        ).get("player_knowledge_rules", [])
        validate_knowledge_rules(knowledge_rules)

        player_knowledge = replay_player_knowledge(
            manifest["player"],
            entries,
            knowledge_rules,
        )
        dump(
            root / "player_knowledge.json",
            player_knowledge,
        )
    else:
        knowledge_rules = []
        player_knowledge = None

    if (
        evaluator_max_tokens is not None
        or evaluator_attempts is not None
        or evaluator_timeout is not None
        or evaluator_temperature is not None
        or evaluator_top_p is not None
        or evaluator_top_k is not None
    ):
        current = manifest.get(
            "runtime", {}
        ).get("semantic_evaluator")

        if current is None or current.get("mode") != "llm":
            raise ValueError(
                "Evaluator overrides require an LLM evaluator"
            )

        candidate = deepcopy(current)

        if evaluator_max_tokens is not None:
            if (
                type(evaluator_max_tokens) is not int
                or not 1 <= evaluator_max_tokens <= 131072
            ):
                raise ValueError("Invalid evaluator max_tokens")
            candidate["sampling"]["max_tokens"] = evaluator_max_tokens

        if evaluator_attempts is not None:
            candidate["max_attempts"] = evaluator_attempts

        if evaluator_timeout is not None:
            if (
                type(evaluator_timeout) is not int
                or not 1 <= evaluator_timeout <= 3600
            ):
                raise ValueError("Invalid evaluator timeout")
            candidate["backend"]["timeout"] = evaluator_timeout

        for value, key in (
            (evaluator_temperature, "temperature"),
            (evaluator_top_p, "top_p"),
            (evaluator_top_k, "top_k"),
        ):
            if value is not None:
                candidate["sampling"][key] = value

        validate_evaluator_config(candidate)

        if candidate != current:
            append(events_path, {
                "type": "evaluator_config_override_v1",
                "previous": current,
                "updated": candidate,
                "timestamp": _now(),
            })
            manifest["runtime"]["semantic_evaluator"] = candidate
            dump(manifest_path, manifest)

    from .evaluator_runtime import resolve_evaluator_config
    evaluator_settings = resolve_evaluator_config(
        manifest.get("runtime", {}).get("semantic_evaluator"),
        manifest["backend"],
    )

    # Archived evaluator proposals must be validated against
    # the original goal state, not the approved live state.
    evaluation_base_state = (
        initial_state(load(root / "ground_truth.json"))
        if evaluator_settings is not None else None
    )
    if evaluator_settings is not None:
        if not controlled_player:
            raise ValueError(
                "Archived evaluator requires controlled_player"
            )
        player_knowledge = reconcile_evaluations(
            root, manifest["player"], entries,
            evaluation_base_state,
            knowledge_rules, evaluator_settings,
            knowledge_only=manifest.get(
                "runtime", {}
            ).get("goal_criteria") is not None,
        )

    goal_rules = manifest.get(
        "runtime", {}
    ).get("goal_approval_rules", [])

    criteria_definition = manifest.get(
        "runtime", {}
    ).get("goal_criteria")

    if criteria_definition is not None and (
        not controlled_player
        or evaluator_settings is None
        or goal_rules
    ):
        raise ValueError("Invalid archived goal_criteria configuration")

    if goal_rules and evaluator_settings is None:
        raise ValueError(
            "Archived goal approval requires an evaluator"
        )

    if criteria_definition is not None:
        from .criteria_runtime import refresh_criteria
        player_state, _, _ = refresh_criteria(
            root, criteria_definition,
            entries, evaluator_settings,
        )
    elif goal_rules:
        player_state, _ = refresh_goal_progress(
            root, entries, goal_rules
        )

    empty_retries = manifest.get(
        "runtime", {}
    ).get("empty_response_retries", 0)

    sampling = manifest["sampling"]
    player_sampling = manifest["player_sampling"]
    prompt_format = manifest["prompt_format"]
    player_format = manifest["player_prompt_format"]

    try:
        for turn in range(completed + 1, target + 1):
            emit(f"Turn {turn:03}/{target:03} | START (resume)")
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
                    render_controlled_player_request(
                        player_messages,
                        player_state,
                        player_knowledge,
                        load(root / "criteria_progress.json")
                        if criteria_definition is not None
                        and (root / "criteria_progress.json").exists()
                        else None,
                    ) if controlled_player else player_messages,
                    player_sampling,
                    player_format,
                    root,
                    turn,
                    "player",
                    manifest.get("player_reasoning_effort"),
                    empty_response_retries=empty_retries,
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
                render_st_npc_request(
                    npc_messages,
                    npc_renderer,
                ) if npc_renderer is not None else npc_messages,
                sampling,
                prompt_format,
                root,
                turn,
                "npc",
                manifest.get("npc_reasoning_effort"),
                empty_response_retries=empty_retries,
            )

            entry = make_observation(
                turn, "npc", npc_text
            )
            append(root / "observations.jsonl", entry)
            entries.append(entry)

            if evaluator_settings is not None:
                player_knowledge = reconcile_evaluations(
                    root, manifest["player"], entries,
                    evaluation_base_state, knowledge_rules,
                    evaluator_settings,
                    knowledge_only=criteria_definition is not None,
                )
                if criteria_definition is not None:
                    from .criteria_runtime import refresh_criteria
                    player_state, _, _ = refresh_criteria(
                        root, criteria_definition,
                        entries, evaluator_settings,
                    )
                elif goal_rules:
                    player_state, _ = refresh_goal_progress(
                        root, entries, goal_rules
                    )
            elif controlled_player:
                player_knowledge = update_player_knowledge(
                    player_knowledge,
                    turn,
                    npc_text,
                    knowledge_rules,
                )
                dump(
                    root / "player_knowledge.json",
                    player_knowledge,
                )

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
            emit(f"Turn {turn:03}/{target:03} | COMPLETE")

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
    parser.add_argument("--evaluator-max-tokens", type=int)
    parser.add_argument("--evaluator-attempts", type=int)
    parser.add_argument("--evaluator-timeout", type=int)
    parser.add_argument("--evaluator-temperature", type=float)
    parser.add_argument("--evaluator-top-p", type=float)
    parser.add_argument("--evaluator-top-k", type=int)
    args = parser.parse_args()

    print(resume_simulation(
        args.run,
        args.turns,
        evaluator_max_tokens=args.evaluator_max_tokens,
        evaluator_attempts=args.evaluator_attempts,
        evaluator_timeout=args.evaluator_timeout,
        evaluator_temperature=args.evaluator_temperature,
        evaluator_top_p=args.evaluator_top_p,
        evaluator_top_k=args.evaluator_top_k,
    ))


if __name__ == "__main__":
    main()
