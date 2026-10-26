"""Versioned semantic criterion proposals with archived evidence.

Validation proves source provenance, not semantic correctness.
No goal status is changed by this module.
"""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json

from .goal_criteria import validate_goal_criteria


SCHEMA_VERSION = 1
EVENT_TYPE = "semantic_criteria_v1"


def definition_hash(definition):
    encoded = json.dumps(
        definition,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def npc_sources(observations):
    if not isinstance(observations, list):
        raise ValueError("Observations must be an array")

    sources = {}

    for item in observations:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid observation")

        if item.get("role") != "npc":
            continue

        turn = item.get("turn")
        text = item.get("text")

        if (
            type(turn) is not int
            or turn < 1
            or not isinstance(text, str)
            or turn in sources
        ):
            raise ValueError("Invalid NPC source")

        sources[turn] = text

    return sources


def validate_semantic_proposal(
    proposal, definition, state, observations,
    current_turn,
):
    """Validate criterion IDs and source provenance.

    Semantic correctness belongs to the evaluator, not a
    deterministic substring check.
    """
    validate_goal_criteria(definition, state)
    sources = npc_sources(observations)

    if (
        type(current_turn) is not int
        or current_turn not in sources
    ):
        raise ValueError("Unknown evaluation turn")

    if (
        not isinstance(proposal, Mapping)
        or set(proposal) != {
            "schema_version", "criterion_updates"
        }
        or type(proposal["schema_version"]) is not int
        or proposal["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Invalid semantic proposal schema")

    updates = proposal["criterion_updates"]
    if not isinstance(updates, list) or len(updates) > 100:
        raise ValueError("Invalid criterion updates")

    clean = []
    seen = set()

    for item in updates:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid criterion update")

        goal_id = item.get("goal_id")
        criterion_id = item.get("criterion_id")
        status = item.get("status")

        if (
            not isinstance(goal_id, str)
            or goal_id not in definition["goals"]
            or not isinstance(criterion_id, str)
        ):
            raise ValueError("Unknown criterion target")

        valid_ids = {
            criterion["id"]
            for criterion in definition["goals"][goal_id]["criteria"]
        }

        key = (goal_id, criterion_id)
        if criterion_id not in valid_ids or key in seen:
            raise ValueError("Unknown or duplicate criterion")

        seen.add(key)

        if status not in {
            "satisfied", "partial", "not_satisfied"
        }:
            raise ValueError("Unsupported criterion status")

        # Existing archived format. It remains readable so
        # a failed run does not require a journal migration.
        if set(item) == {
            "goal_id", "criterion_id", "status", "evidence"
        }:
            if status != "satisfied":
                raise ValueError("Invalid legacy criterion status")

            evidence = item["evidence"]
            if not isinstance(evidence, list) or not (
                1 <= len(evidence) <= 20
            ):
                raise ValueError("Invalid criterion evidence")

            references = []
            seen_references = set()

            for reference in evidence:
                if (
                    not isinstance(reference, Mapping)
                    or set(reference) != {"turn", "quote"}
                ):
                    raise ValueError("Invalid evidence reference")

                turn = reference["turn"]
                quote = reference["quote"]

                if (
                    type(turn) is not int
                    or turn < 1
                    or turn > current_turn
                    or turn not in sources
                    or not isinstance(quote, str)
                    or not quote.strip()
                    or quote not in sources[turn]
                ):
                    raise ValueError(
                        "Criterion evidence source mismatch"
                    )

                signature = (turn, quote)
                if signature in seen_references:
                    raise ValueError("Duplicate criterion evidence")
                seen_references.add(signature)
                references.append({"turn": turn, "quote": quote})

            clean.append({
                "goal_id": goal_id,
                "criterion_id": criterion_id,
                "status": status,
                "evidence": references,
            })
            continue

        if set(item) != {
            "goal_id", "criterion_id", "status",
            "source_turns", "reason"
        }:
            raise ValueError("Invalid semantic criterion fields")

        turns = item["source_turns"]
        reason = item["reason"]

        minimum_sources = (
            0 if status == "not_satisfied" else 1
        )

        if (
            not isinstance(turns, list)
            or not minimum_sources <= len(turns) <= 20
            or any(type(turn) is not int for turn in turns)
            or len(set(turns)) != len(turns)
            or any(
                turn < 1
                or turn > current_turn
                or turn not in sources
                for turn in turns
            )
        ):
            raise ValueError("Invalid source turns")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Missing semantic reason")

        clean.append({
            "goal_id": goal_id,
            "criterion_id": criterion_id,
            "status": status,
            "source_turns": list(turns),
            "reason": reason.strip(),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "criterion_updates": clean,
    }


def make_semantic_event(
    turn, proposal, definition, state,
    observations, evaluator_id, goal_id=None,
):
    if goal_id is not None and (
        not isinstance(goal_id, str)
        or goal_id not in definition["goals"]
    ):
        raise ValueError("Unknown assessed goal")

    if goal_id is not None:
        for update in proposal.get("criterion_updates", []):
            if update.get("goal_id") != goal_id:
                raise ValueError("Cross-goal criterion update")

    if not isinstance(evaluator_id, str) or not (
        evaluator_id.strip()
    ):
        raise ValueError("Invalid evaluator ID")

    validated = validate_semantic_proposal(
        proposal, definition, state, observations, turn
    )

    sources = npc_sources(observations)

    event = {
        "type": EVENT_TYPE,
        "turn": turn,
        "evaluator_id": evaluator_id,
        "definition_sha256": definition_hash(definition),
        "npc_response_sha256": sha256(
            sources[turn].encode("utf-8")
        ).hexdigest(),
        "result": validated,
    }

    if goal_id is not None:
        event["goal_id"] = goal_id

    return event


def replay_semantic_criteria(
    definition, state, observations, events,
):
    """Replay audited proposals, preserving partial progress.

    The result contains proposals, not approved goal changes.
    """
    definition = validate_goal_criteria(definition, state)

    if not isinstance(events, list):
        raise ValueError("Events must be an array")

    progress = {
        goal_id: {
            "criteria": {
                criterion["id"]: {
                    "description": criterion["description"],
                    "proposed_satisfied": False,
                    "evidence": [],
                }
                for criterion in entry["criteria"]
            },
            "proposed_satisfied": False,
        }
        for goal_id, entry in definition["goals"].items()
    }

    seen_turns = set()

    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("Invalid semantic event")

        if event.get("type") != EVENT_TYPE:
            continue

        required = {
            "type", "turn", "evaluator_id",
            "definition_sha256", "npc_response_sha256",
            "result",
        }
        if set(event) not in (required, required | {"goal_id"}):
            raise ValueError("Invalid semantic event fields")

        turn = event["turn"]
        goal_id = event.get("goal_id")

        if type(turn) is not int or turn < 1:
            raise ValueError("Invalid semantic turn")

        # Legacy event: evaluated every goal at this turn.
        # New event: evaluated exactly one goal at this turn.
        if goal_id is None:
            if turn in seen_turns or any(
                isinstance(key, tuple) and key[0] == turn
                for key in seen_turns
            ):
                raise ValueError("Duplicate semantic turn")
            seen_turns.add(turn)
        else:
            if turn in seen_turns or (turn, goal_id) in seen_turns:
                raise ValueError("Duplicate semantic goal assessment")
            if goal_id not in definition["goals"]:
                raise ValueError("Unknown assessed goal")
            seen_turns.add((turn, goal_id))

        expected = make_semantic_event(
            turn,
            event["result"],
            definition,
            state,
            observations,
            event["evaluator_id"],
            goal_id=goal_id,
        )

        if expected != event:
            raise ValueError("Semantic event provenance mismatch")

        for item in event["result"]["criterion_updates"]:
            record = progress[
                item["goal_id"]
            ]["criteria"][item["criterion_id"]]

            if item["status"] == "satisfied":
                record["proposed_satisfied"] = True

            if "evidence" in item:
                references = item["evidence"]
            else:
                references = [
                    {
                        "turn": turn,
                        "reason": item["reason"],
                        "status": item["status"],
                    }
                    for turn in item["source_turns"]
                ]

            for reference in references:
                if reference not in record["evidence"]:
                    record["evidence"].append(
                        deepcopy(reference)
                    )

    for goal in progress.values():
        goal["proposed_satisfied"] = all(
            item["proposed_satisfied"]
            for item in goal["criteria"].values()
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "goals": progress,
    }
