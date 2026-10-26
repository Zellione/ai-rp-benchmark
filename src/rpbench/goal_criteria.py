"""Versioned, deterministic evidence tracking for RP goals.

Criterion satisfaction is not authorization to complete a goal.
The caller must supply already validated evaluation events.
"""

from collections.abc import Mapping
from copy import deepcopy

SCHEMA_VERSION = 1


def validate_goal_criteria(definition, state):
    if not isinstance(definition, Mapping):
        raise ValueError("Goal criteria must be an object")

    if set(definition) != {"schema_version", "goals"}:
        raise ValueError("Invalid goal criteria fields")

    if type(definition["schema_version"]) is not int or (
        definition["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported goal criteria schema")

    goals = definition["goals"]
    known = state.get("goals") if isinstance(state, Mapping) else None

    if not isinstance(goals, Mapping):
        raise ValueError("Criteria goals must be an object")
    if not isinstance(known, Mapping):
        raise ValueError("Invalid goal state")

    for goal_id, definition_entry in goals.items():
        if goal_id not in known:
            raise ValueError(f"Unknown criterion goal: {goal_id}")

        if (
            not isinstance(definition_entry, Mapping)
            or set(definition_entry) != {"criteria"}
        ):
            raise ValueError("Invalid goal criteria entry")

        criteria = definition_entry["criteria"]
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Goal criteria must be nonempty")

        seen = set()
        for criterion in criteria:
            if (
                not isinstance(criterion, Mapping)
                or set(criterion) != {
                    "id", "description", "evidence_contains"
                }
            ):
                raise ValueError("Invalid criterion")

            for field in (
                "id", "description", "evidence_contains"
            ):
                value = criterion[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Invalid criterion {field}")

            if criterion["id"] in seen:
                raise ValueError("Duplicate criterion ID")

            seen.add(criterion["id"])

    return deepcopy(definition)


def replay_goal_criteria(definition, state, evaluations):
    """Compute evidence-backed progress without changing goal states.

    Evaluations are expected to have been validated against
    archived NPC responses by read_evaluations().
    """
    definition = validate_goal_criteria(definition, state)

    if not isinstance(evaluations, Mapping):
        raise ValueError("Evaluations must be a mapping")

    progress = {}

    for goal_id, entry in definition["goals"].items():
        progress[goal_id] = {
            "criteria": {
                item["id"]: {
                    "description": item["description"],
                    "satisfied": False,
                    "evidence": [],
                }
                for item in entry["criteria"]
            },
            "satisfied": False,
        }

    for turn in sorted(evaluations):
        if type(turn) is not int or turn < 1:
            raise ValueError("Invalid evaluation turn")

        event = evaluations[turn]
        if not isinstance(event, Mapping) or event.get("turn") != turn:
            raise ValueError("Evaluation turn mismatch")

        result = event.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("Invalid evaluation result")

        knowledge = result.get("knowledge_updates")
        goals = result.get("goal_updates")
        if not isinstance(knowledge, list) or not isinstance(goals, list):
            raise ValueError("Invalid evaluation updates")

        # Both evidence types are already normalized by the
        # authoritative evaluation journal.
        evidence_items = knowledge + goals

        for item in evidence_items:
            if not isinstance(item, Mapping):
                raise ValueError("Invalid evidence entry")
            quote = item.get("evidence")
            if not isinstance(quote, str):
                raise ValueError("Invalid evidence quote")

            for goal_id, definition_entry in definition["goals"].items():
                for criterion in definition_entry["criteria"]:
                    needle = criterion["evidence_contains"]
                    if needle.casefold() not in quote.casefold():
                        continue

                    record = progress[goal_id]["criteria"][
                        criterion["id"]
                    ]

                    if not any(
                        previous["quote"] == quote
                        for previous in record["evidence"]
                    ):
                        record["evidence"].append({
                            "turn": turn,
                            "quote": quote,
                        })

    for entry in progress.values():
        for criterion in entry["criteria"].values():
            criterion["satisfied"] = bool(criterion["evidence"])

        entry["satisfied"] = all(
            criterion["satisfied"]
            for criterion in entry["criteria"].values()
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "goals": progress,
    }
