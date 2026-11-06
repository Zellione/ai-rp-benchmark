"""Prepare a non-destructive RPBench false-positive restart branch."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .core import load
from .criteria_progress import advance_criteria_goals
from .false_positive_audit import (
    digest,
    find_completions,
    validate_verdict,
)
from .false_positive_branch import (
    correct_semantic_events,
    rejected_criteria,
)
from .goal_criteria import required_goal_summary, validate_goal_criteria
from .observations import (
    read_jsonl,
    read_observations,
    validate_sequence,
)
from .player_controller import select_player_goal
from .resume import _render_pair
from .semantic_criteria import EVENT_TYPE
from .state import initial_state


def write_json(file, value):
    file.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(file, values):
    file.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def select_confirmed_failure(
    audit_root, source, manifest, definition,
    observations, completions,
):
    """Require a validated verdict for every earlier completion."""
    for candidate in completions:
        turn = candidate["turn"]
        goal_id = candidate["goal_id"]
        folder = audit_root / f"turn-{turn:03d}-{goal_id}"

        verdict_file = folder / "verdict.json"
        identity_file = folder / "identity.json"

        if not verdict_file.is_file() or not identity_file.is_file():
            raise ValueError(
                f"Missing chronological audit: {folder}"
            )

        identity = load(identity_file)
        result = load(verdict_file)

        if identity.get("source_run") != str(source):
            raise ValueError("Audit belongs to another source run")

        source_hash = sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest()

        if identity.get("source_manifest_sha256") != source_hash:
            raise ValueError("Source manifest changed since audit")

        if result.get("identity_sha256") != digest(identity):
            raise ValueError("Audit identity mismatch")

        prefix = [
            item for item in observations
            if item["turn"] <= turn
        ]

        verdict = validate_verdict(
            result["verdict"], goal_id, definition, prefix
        )

        statuses = {
            item["status"] for item in verdict["criteria"]
        }

        expected_class = (
            "uncertain" if "uncertain" in statuses
            else "valid" if statuses == {"satisfied"}
            else "false_positive"
        )

        if result.get("classification") != expected_class:
            raise ValueError("Audit classification mismatch")

        if expected_class == "uncertain":
            raise ValueError(
                f"Uncertain audit at turn {turn}; no automatic rollback"
            )

        if expected_class == "false_positive":
            return {
                "turn": turn,
                "goal_id": goal_id,
                "rejected": rejected_criteria(verdict),
                "audit_file": str(verdict_file),
                "synthetic": False,
            }

    raise ValueError("No confirmed false positive in audited run")


def prepare_branch(
    source, destination, *,
    audit_root=None,
    synthetic_turn=None,
    synthetic_goal=None,
    synthetic_criterion=None,
):
    source = Path(source).resolve()
    destination = Path(destination).resolve()

    if not source.is_dir():
        raise ValueError("Source run missing")

    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside source run")

    if destination.exists():
        raise ValueError("Destination already exists")

    manifest = load(source / "manifest.json")

    if (
        manifest.get("status") != "complete"
        or manifest.get("type") != "goal_driven_selfplay"
    ):
        raise ValueError("Source must be a complete self-play run")

    ground_truth = load(source / "ground_truth.json")
    initial = initial_state(ground_truth)
    definition = validate_goal_criteria(
        manifest["runtime"]["goal_criteria"], initial
    )

    observations = read_observations(source)
    events = read_jsonl(source / "events.jsonl")
    completions = find_completions(
        definition, initial, observations, events
    )

    if audit_root is not None:
        if any(x is not None for x in (
            synthetic_turn, synthetic_goal, synthetic_criterion
        )):
            raise ValueError("Cannot combine real and synthetic modes")

        failure = select_confirmed_failure(
            Path(audit_root).resolve(),
            source, manifest, definition,
            observations, completions,
        )
    else:
        if (
            type(synthetic_turn) is not int
            or not isinstance(synthetic_goal, str)
            or not isinstance(synthetic_criterion, str)
        ):
            raise ValueError(
                "Provide --audit-root or an explicit synthetic target"
            )

        candidate = {
            "turn": synthetic_turn,
            "goal_id": synthetic_goal,
        }

        if candidate not in completions:
            raise ValueError("Synthetic target is not a goal completion")

        failure = {
            **candidate,
            "rejected": {synthetic_criterion},
            "audit_file": None,
            "synthetic": True,
        }

    cutoff = failure["turn"]
    goal_id = failure["goal_id"]

    prefix = [
        deepcopy(item)
        for item in observations
        if item["turn"] <= cutoff
    ]

    validate_sequence(prefix)

    if len(prefix) != cutoff * 2:
        raise ValueError("Incomplete observation prefix")

    # The new journal contains no later turns.
    prefix_events = [
        deepcopy(item)
        for item in events
        if type(item.get("turn")) is int
        and item["turn"] <= cutoff
    ]

    corrected_events, changes = correct_semantic_events(
        prefix_events,
        definition,
        initial,
        prefix,
        cutoff,
        goal_id,
        failure["rejected"],
    )

    semantic = [
        item for item in corrected_events
        if item.get("type") == EVENT_TYPE
    ]

    state, progress, _ = advance_criteria_goals(
        initial, definition, prefix, semantic
    )

    if state["goals"][goal_id]["status"] == "completed":
        raise ValueError(
            "Correction did not reopen falsely completed goal"
        )

    next_goal = select_player_goal(state, definition)

    if next_goal is None or next_goal["id"] != goal_id:
        raise ValueError(
            "Player would not return to corrected goal; "
            "manual review required"
        )

    progress["overall_goal"] = required_goal_summary(
        definition, state
    )

    title = manifest["scenario"].get(
        "title", manifest["scenario"].get("id")
    )

    transcript = (
        f"# {title}\n\n"
        f"Player: {manifest['player']['name']}\n\n"
        "Mode: Goal-driven self-play\n\n"
    )

    opening = manifest["scenario"].get("npc_opening")

    if opening is not None:
        transcript += f"## NPC Opening\n\n{opening}\n\n"

    for turn in range(1, cutoff + 1):
        player = prefix[2 * (turn - 1)]
        npc = prefix[2 * (turn - 1) + 1]

        transcript += _render_pair(
            turn,
            manifest["player"]["name"],
            player["text"],
            npc["text"],
        )

    if not (source / "conversation.md").read_text(
        encoding="utf-8"
    ).startswith(transcript):
        raise ValueError("Transcript prefix mismatch")

    branch_manifest = deepcopy(manifest)
    branch_manifest["completed_turns"] = cutoff
    branch_manifest["status"] = "failed"
    branch_manifest["error"] = (
        "Prepared false-positive recovery branch; resume required"
    )
    branch_manifest.pop("finished_utc", None)

    provenance = {
        "schema_version": 1,
        "source_run": str(source),
        "source_manifest_sha256": sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
        "preserved_through_turn": cutoff,
        "restart_player_turn": cutoff + 1,
        "corrected_goal": goal_id,
        "rejected_criteria": sorted(failure["rejected"]),
        "corrections": changes,
        "audit_file": failure["audit_file"],
        "synthetic": failure["synthetic"],
        "benchmark_eligible": not failure["synthetic"],
    }

    # All checks above happen before creating the destination.
    destination.mkdir(parents=True, exist_ok=False)

    write_json(destination / "manifest.json", branch_manifest)
    write_json(destination / "ground_truth.json", ground_truth)
    write_json(destination / "state.json", state)
    write_json(destination / "criteria_progress.json", progress)
    write_json(destination / "branch_origin.json", provenance)

    write_jsonl(destination / "observations.jsonl", prefix)
    write_jsonl(destination / "events.jsonl", corrected_events)

    (destination / "conversation.md").write_text(
        transcript, encoding="utf-8"
    )

    # Verify persisted data by replaying the corrected journal.
    persisted = read_jsonl(destination / "events.jsonl")
    persisted_semantic = [
        item for item in persisted
        if item.get("type") == EVENT_TYPE
    ]

    restored_state, _, _ = advance_criteria_goals(
        initial, definition, prefix, persisted_semantic
    )

    if restored_state != state:
        raise ValueError("Persisted branch replay mismatch")

    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-root")
    parser.add_argument("--synthetic-turn", type=int)
    parser.add_argument("--synthetic-goal")
    parser.add_argument("--synthetic-criterion")
    args = parser.parse_args()

    result = prepare_branch(
        args.run,
        args.out,
        audit_root=args.audit_root,
        synthetic_turn=args.synthetic_turn,
        synthetic_goal=args.synthetic_goal,
        synthetic_criterion=args.synthetic_criterion,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
