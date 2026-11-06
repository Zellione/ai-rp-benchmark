"""Reconstruct corrected semantic events for a false-positive branch.

The source event journal is never modified.
"""

from copy import deepcopy

from .semantic_criteria import EVENT_TYPE, make_semantic_event


def rejected_criteria(verdict):
    """Return criteria independently judged not fully satisfied."""
    if not isinstance(verdict, dict):
        raise ValueError("Invalid audit verdict")

    items = verdict.get("criteria")
    if not isinstance(items, list) or not items:
        raise ValueError("Missing audit criteria")

    rejected = set()
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid criterion verdict")

        identifier = item.get("criterion_id")
        status = item.get("status")

        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
        ):
            raise ValueError("Duplicate or invalid criterion")

        if status not in {
            "satisfied", "partial", "not_satisfied", "uncertain"
        }:
            raise ValueError("Invalid audit status")

        if status == "uncertain":
            raise ValueError(
                "Uncertain audit cannot authorize rollback"
            )

        seen.add(identifier)

        if status != "satisfied":
            rejected.add(identifier)

    if not rejected:
        raise ValueError(
            "Audit contains no disproven criteria"
        )

    return rejected


def correct_semantic_events(
    events,
    definition,
    initial_state,
    observations,
    completion_turn,
    goal_id,
    rejected,
):
    """Remove disproven satisfied updates through completion_turn.

    Reconstruct each modified event with valid source hashes.
    Preserve all unrelated event data.
    """
    if (
        type(completion_turn) is not int
        or completion_turn < 1
        or goal_id not in definition["goals"]
        or not isinstance(rejected, set)
        or not rejected
    ):
        raise ValueError("Invalid rollback target")

    valid_ids = {
        item["id"]
        for item in definition["goals"][goal_id]["criteria"]
    }

    if not rejected <= valid_ids:
        raise ValueError("Unknown rejected criterion")

    corrected = []
    changes = []

    for event in events:
        updated = deepcopy(event)

        if (
            event.get("type") != EVENT_TYPE
            or event["turn"] > completion_turn
        ):
            corrected.append(updated)
            continue

        updates = updated["result"]["criterion_updates"]

        remaining = [
            item
            for item in updates
            if not (
                item["goal_id"] == goal_id
                and item["criterion_id"] in rejected
                and item["status"] == "satisfied"
            )
        ]

        if len(remaining) == len(updates):
            corrected.append(updated)
            continue

        removed = [
            item["criterion_id"]
            for item in updates
            if item not in remaining
        ]

        replacement = make_semantic_event(
            event["turn"],
            {
                "schema_version": 1,
                "criterion_updates": remaining,
            },
            definition,
            initial_state,
            observations,
            event["evaluator_id"],
            goal_id=event.get("goal_id"),
        )

        corrected.append(replacement)

        changes.append({
            "turn": event["turn"],
            "goal_id": goal_id,
            "removed_criteria": removed,
        })

    if not changes:
        raise ValueError(
            "No matching satisfied events to correct"
        )

    return corrected, changes
