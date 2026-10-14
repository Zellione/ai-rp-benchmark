"""Conservative, event-sourced RP state tracking."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone


VALID_GOAL_STATES = {
    "pending", "active", "completed", "failed"
}

VALID_EVENT_TYPES = {
    "knowledge_acquired",
    "relationship_changed",
    "goal_status_changed",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_ground_truth(player, character_cards):
    characters = {}

    for card in character_cards:
        character_id = card["id"]

        if character_id in characters:
            raise ValueError(
                f"Duplicate character: {character_id}"
            )

        characters[character_id] = {
            "name": card["name"],
            "appearance": deepcopy(card["appearance"]),
            "personality": deepcopy(card["personality"]),
            "speech": deepcopy(card["speech"]),
            "invariants": deepcopy(card["invariants"]),
            "initial_knowledge": deepcopy(
                card["knowledge"]
            ),
            "private_knowledge": deepcopy(
                card["private_knowledge"]
            ),
            "relationships": deepcopy(
                card.get("relationships", {})
            ),
        }

    goals = {
        f"subgoal-{i:02}": {
            "description": description,
            "initial_status": "pending",
        }
        for i, description in enumerate(
            player["subgoals"], 1
        )
    }

    return {
        "schema_version": 1,
        "player": {
            "name": player["name"],
            "main_goal": player["main_goal"],
            "subgoals": goals,
        },
        "characters": characters,
    }


def initial_state(ground_truth):
    return {
        "schema_version": 1,
        "last_confirmed_event": 0,
        "character_knowledge": {
            cid: {
                "known_facts": list(
                    data["initial_knowledge"]
                ),
                "private_facts": list(
                    data["private_knowledge"]
                ),
            }
            for cid, data in ground_truth[
                "characters"
            ].items()
        },
        "relationships": {
            cid: deepcopy(
                data["relationships"]
            )
            for cid, data in ground_truth[
                "characters"
            ].items()
        },
        "goals": {
            gid: {
                "description": data["description"],
                "status": data["initial_status"],
            }
            for gid, data in ground_truth[
                "player"
            ]["subgoals"].items()
        },
    }


def make_observation(turn, role, text):
    if role not in {"player", "npc"}:
        raise ValueError("Invalid observation role")
    if type(turn) is not int or turn <= 0:
        raise ValueError("Invalid turn")

    return {
        "type": "dialogue_observation",
        "turn": turn,
        "role": role,
        "text": text,
        "timestamp": utc_now(),
        "verified": False,
    }


def apply_confirmed_event(state, event):
    """Return new state. Never mutate input or infer from text."""
    if event.get("type") not in VALID_EVENT_TYPES:
        raise ValueError("Unsupported state event")
    if event.get("verified") is not True:
        raise ValueError("Event is not verified")
    if event.get("verified_by") not in {
        "human", "rule", "adjudicator"
    }:
        raise ValueError("Missing verification authority")
    if type(event.get("event_id")) is not int:
        raise ValueError("Event ID must be an integer")
    if event["event_id"] != (
        state["last_confirmed_event"] + 1
    ):
        raise ValueError("Out-of-order state event")

    updated = deepcopy(state)
    kind = event["type"]

    if kind == "knowledge_acquired":
        cid = event["character_id"]
        fact = event["fact"]

        if cid not in updated["character_knowledge"]:
            raise ValueError("Unknown character")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("Invalid fact")

        facts = updated[
            "character_knowledge"
        ][cid]["known_facts"]

        if fact not in facts:
            facts.append(fact)

    elif kind == "relationship_changed":
        cid = event["character_id"]
        other = event["target"]
        description = event["description"]

        if cid not in updated["relationships"]:
            raise ValueError("Unknown character")
        if (
            not isinstance(other, str)
            or not other.strip()
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise ValueError("Invalid relationship")

        updated["relationships"][cid][
            other
        ] = description

    elif kind == "goal_status_changed":
        gid = event["goal_id"]
        status = event["status"]

        if gid not in updated["goals"]:
            raise ValueError("Unknown goal")
        if status not in VALID_GOAL_STATES:
            raise ValueError("Unknown goal status")

        updated["goals"][gid]["status"] = status

    updated["last_confirmed_event"] = event["event_id"]
    return updated
