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
    """Validate shape and exact source provenance only."""
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
        if not isinstance(item, Mapping) or set(item) != {
            "goal_id", "criterion_id", "status", "evidence"
        }:
            raise ValueError("Invalid criterion update")

        goal_id = item["goal_id"]
        criterion_id = item["criterion_id"]

        if (
            not isinstance(goal_id, str)
            or goal_id not in definition["goals"]
            or not isinstance(criterion_id, str)
        ):
            raise ValueError("Unknown criterion target")

        known = {
            criterion["id"]
            for criterion in definition["goals"][
                goal_id
            ]["criteria"]
        }

        key = (goal_id, criterion_id)

        if criterion_id not in known or key in seen:
            raise ValueError("Unknown or duplicate criterion")

        seen.add(key)

        if item["status"] != "satisfied":
            raise ValueError("Unsupported criterion status")

        evidence = item["evidence"]
        if not isinstance(evidence, list) or not (
            1 <= len(evidence) <= 20
        ):
            raise ValueError("Invalid criterion evidence")

        clean_evidence = []
        seen_quotes = set()

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
            if signature in seen_quotes:
                raise ValueError("Duplicate criterion evidence")

            seen_quotes.add(signature)
            clean_evidence.append({
                "turn": turn,
                "quote": quote,
            })

        clean.append({
            "goal_id": goal_id,
            "criterion_id": criterion_id,
            "status": "satisfied",
            "evidence": clean_evidence,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "criterion_updates": clean,
    }


def make_semantic_event(
    turn, proposal, definition, state,
    observations, evaluator_id,
):
    if not isinstance(evaluator_id, str) or not (
        evaluator_id.strip()
    ):
        raise ValueError("Invalid evaluator ID")

    validated = validate_semantic_proposal(
        proposal, definition, state, observations, turn
    )

    sources = npc_sources(observations)

    return {
        "type": EVENT_TYPE,
        "turn": turn,
        "evaluator_id": evaluator_id,
        "definition_sha256": definition_hash(definition),
        "npc_response_sha256": sha256(
            sources[turn].encode("utf-8")
        ).hexdigest(),
        "result": validated,
    }


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

        if set(event) != {
            "type", "turn", "evaluator_id",
            "definition_sha256", "npc_response_sha256",
            "result",
        }:
            raise ValueError("Invalid semantic event fields")

        turn = event["turn"]
        if type(turn) is not int or turn in seen_turns:
            raise ValueError("Duplicate or invalid semantic turn")

        seen_turns.add(turn)

        expected = make_semantic_event(
            turn,
            event["result"],
            definition,
            state,
            observations,
            event["evaluator_id"],
        )

        if expected != event:
            raise ValueError("Semantic event provenance mismatch")

        for item in event["result"]["criterion_updates"]:
            record = progress[
                item["goal_id"]
            ]["criteria"][item["criterion_id"]]

            record["proposed_satisfied"] = True

            for reference in item["evidence"]:
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
