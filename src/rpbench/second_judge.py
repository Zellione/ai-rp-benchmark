"""Independent, resumable offline audit of archived criteria decisions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .core import append, dump, load
from .criteria_runtime import _generate
from .goal_criteria import validate_goal_criteria
from .observations import read_observations, read_jsonl
from .semantic_criteria import (
    make_semantic_event,
    replay_semantic_criteria,
)
from .state import initial_state


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def audit(
    run, output, model, goals=None, limit=None,
    dry_run=False, max_tokens=16384,
    base_url="https://nano-gpt.com/api/subscription",
):
    run = Path(run).resolve()
    output = Path(output).resolve()

    manifest = load(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be audited")

    definition = manifest["runtime"]["goal_criteria"]
    state = initial_state(load(run / "ground_truth.json"))
    definition = validate_goal_criteria(definition, state)

    observations = read_observations(run)
    source_events = [
        event for event in read_jsonl(run / "events.jsonl")
        if event.get("type") == "semantic_criteria_v1"
    ]

    selected = [
        event for event in source_events
        if "goal_id" in event
        and (goals is None or event["goal_id"] in goals)
    ]

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]

    if not selected:
        raise ValueError("No matching archived goal evaluations")

    source_identity = {
        "manifest_sha256": digest(manifest),
        "observations_sha256": digest(observations),
        "definition_sha256": digest(definition),
        "primary_events_sha256": digest(source_events),
        "judge_model": model,
        "selected": [
            [event["turn"], event["goal_id"]]
            for event in selected
        ],
        "max_tokens": max_tokens,
        "base_url": base_url,
    }

    if dry_run:
        print("Run:", run)
        print("Model:", model)
        print("Selected evaluations:", len(selected))
        for event in selected:
            print(f"  turn={event['turn']} goal={event['goal_id']}")
        return

    if output == run or run in output.parents:
        raise ValueError("Audit output must be outside source archive")

    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / "identity.json"

    if identity_file.exists():
        if load(identity_file) != source_identity:
            raise ValueError("Audit identity mismatch")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty uninitialized audit directory")
        dump(identity_file, source_identity)

    events_file = output / "events.jsonl"
    audit_events = (
        read_jsonl(events_file)
        if events_file.exists() else []
    )

    valid_keys = {
        (event["turn"], event["goal_id"])
        for event in selected
    }
    seen = set()
    secondary = []

    for event in audit_events:
        if event.get("type") != "semantic_criteria_v1":
            raise ValueError("Unexpected audit journal event")
        key = (event["turn"], event["goal_id"])
        if key not in valid_keys or key in seen:
            raise ValueError("Duplicate or unexpected audit event")
        seen.add(key)
        secondary.append(event)

    settings = {
        "backend": {
            "name": "nanogpt",
            "base_url": base_url,
            "model": model,
            "api_key_env": "NANOGPT_API_KEY",
            "timeout": 300,
        },
        "sampling": {
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    }

    # Replay each selected primary evaluation at its original turn.
    # The secondary judge sees its own earlier assessments only.
    report_rows = []

    for original in selected:
        turn = original["turn"]
        goal_id = original["goal_id"]
        key = (turn, goal_id)
        available = [
            item for item in observations
            if item["turn"] <= turn
        ]

        previous = [
            item for item in secondary
            if item["turn"] < turn
        ]
        progress = replay_semantic_criteria(
            definition, state, available, previous
        )

        original_ids = [
            item["criterion_id"]
            for item in original["result"]["criterion_updates"]
        ]

        pending = [
            criterion_id for criterion_id in original_ids
            if not progress["goals"][goal_id][
                "criteria"
            ][criterion_id]["proposed_satisfied"]
        ]

        if key not in seen and pending:
            if not os.getenv("NANOGPT_API_KEY"):
                raise ValueError("NANOGPT_API_KEY is missing")

            proposal, metadata = _generate(
                definition, progress, available, settings,
                goal_id=goal_id,
                criterion_ids=pending,
            )

            if metadata["finish_reason"] == "length":
                raise ValueError(
                    f"Secondary judge exhausted token budget: {key}"
                )

            proposed_ids = [
                item["criterion_id"]
                for item in proposal["criterion_updates"]
            ]
            if (
                len(proposed_ids) != len(pending)
                or set(proposed_ids) != set(pending)
            ):
                raise ValueError(
                    f"Incomplete secondary decision: {key}"
                )

            event = make_semantic_event(
                turn, proposal, definition, state,
                available, f"offline:{model}",
                goal_id=goal_id,
            )

            append(events_file, event)
            secondary.append(event)
            seen.add(key)

        current = replay_semantic_criteria(
            definition, state, available,
            [
                item for item in secondary
                if item["turn"] <= turn
            ],
        )

        primary_updates = {
            item["criterion_id"]: item["status"]
            for item in original["result"]["criterion_updates"]
        }

        secondary_event = next(
            (
                item for item in secondary
                if (item["turn"], item["goal_id"]) == key
            ),
            None,
        )
        secondary_updates = {
            item["criterion_id"]: item
            for item in (
                secondary_event["result"]["criterion_updates"]
                if secondary_event is not None else []
            )
        }

        for criterion_id, primary_status in primary_updates.items():
            record = current["goals"][goal_id][
                "criteria"
            ][criterion_id]

            if record["proposed_satisfied"]:
                secondary_status = "satisfied"
            else:
                secondary_status = secondary_updates.get(
                    criterion_id, {}
                ).get("status", "not_assessed")

            report_rows.append({
                "turn": turn,
                "goal_id": goal_id,
                "criterion_id": criterion_id,
                "primary_status": primary_status,
                "secondary_status": secondary_status,
                "agreement": primary_status == secondary_status,
                "secondary_reason": secondary_updates.get(
                    criterion_id, {}
                ).get("reason"),
                "secondary_source_turns": secondary_updates.get(
                    criterion_id, {}
                ).get("source_turns", []),
            })

    comparisons = [
        row for row in report_rows
        if row["secondary_status"] != "not_assessed"
    ]
    agreements = sum(row["agreement"] for row in comparisons)

    report = {
        "schema_version": 1,
        "source_run": str(run),
        "judge_model": model,
        "total": len(comparisons),
        "agreements": agreements,
        "disagreements": len(comparisons) - agreements,
        "rows": report_rows,
    }
    dump(output / "agreement.json", report)

    print(
        f"Audit complete: {agreements}/{len(comparisons)} "
        "criterion decisions agree"
    )
    print("Report:", output / "agreement.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--goals", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--base-url",
        default="https://nano-gpt.com/api/subscription",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit(
        args.run, args.out, args.model,
        goals=set(args.goals) if args.goals else None,
        limit=args.limit,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
