"""Minimal, deterministic goal controller for player simulation."""

from collections.abc import Mapping

from .player_knowledge import format_player_knowledge


def select_player_goal(state: Mapping) -> dict | None:
    """Select an active goal, otherwise the first pending goal.

    Goal completion is never inferred from model-generated text.
    Only explicitly maintained goal states influence selection.
    """
    if not isinstance(state, Mapping):
        raise ValueError("Invalid player state")

    goals = state.get("goals")

    if not isinstance(goals, Mapping):
        raise ValueError("Missing goals")

    candidates = []

    for identifier, goal in goals.items():
        if not isinstance(identifier, str) or not isinstance(goal, Mapping):
            raise ValueError("Invalid goal entry")

        status = goal.get("status")
        description = goal.get("description")

        if status not in {
            "pending", "active", "completed", "skipped", "failed"
        }:
            raise ValueError("Invalid goal status")

        if not isinstance(description, str) or not description.strip():
            raise ValueError("Invalid goal description")

        if status in {"active", "pending"}:
            candidates.append({
                "id": identifier,
                "description": description,
                "status": status,
            })

    for status in ("active", "pending"):
        for goal in candidates:
            if goal["status"] == status:
                return goal

    return None


def build_controlled_player_instruction(
    state: Mapping, criteria_progress=None
) -> str:
    """Return goal guidance; never assert unverified success."""
    goal = select_player_goal(state)

    if goal is None:
        return (
            "CONTROLLED PLAYER OBJECTIVE:\n"
            "No open intermediate goals remain. React naturally "
            "without inventing further objectives or claiming "
            "unverified success."
        )

    guidance = (
        "CONTROLLED PLAYER OBJECTIVE:\n"
        f"Current intermediate goal: {goal['description']}\n"
        "Use the actual NPC response to choose your next action.\n"
        "If blocked, investigate, negotiate, or try another approach.\n"
        "Do not force the NPC to cooperate or narrate success.\n"
        "Do not independently mark goals as completed.\n"
        "Keep pursuing the broader goal without rushing."
    )

    if criteria_progress is not None:
        entry = criteria_progress.get(
            "goals", {}
        ).get(goal["id"])

        if entry is not None:
            missing = [
                item["description"]
                for item in entry["criteria"].values()
                if not item["proposed_satisfied"]
            ]
            if missing:
                guidance += (
                    "\nStill unresolved:\n"
                    + "\n".join(f"- {item}" for item in missing)
                )

    return guidance



def render_controlled_player_request(
    messages, state, knowledge=None, criteria_progress=None
):
    """Add current goal guidance to a request without mutating history."""
    if (
        not isinstance(messages, (list, tuple))
        or not messages
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ValueError("Invalid player request history")

    guidance = build_controlled_player_instruction(
        state, criteria_progress
    )

    if knowledge is not None:
        guidance += "\n\n" + format_player_knowledge(knowledge)

    result = [dict(message) for message in messages]
    result[0]["content"] += "\n\n" + guidance
    return result
