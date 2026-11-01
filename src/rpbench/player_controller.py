"""Minimal, deterministic goal controller for player simulation."""

from collections.abc import Mapping

from .player_knowledge import format_player_knowledge


def select_player_goal(
    state: Mapping, definition=None
) -> dict | None:
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
            if definition is not None:
                from .goal_criteria import prerequisites_met
                if not prerequisites_met(definition, state, identifier):
                    continue
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
    state: Mapping, criteria_progress=None, definition=None,
    mode="action",
) -> str:
    """Return goal guidance; never assert unverified success."""
    if mode not in {"action", "narrative"}:
        raise ValueError("Invalid controlled player mode")

    goal = select_player_goal(state, definition)

    if mode == "narrative":
        objective = (
            goal["description"]
            if goal is not None
            else "Continue the story according to established events."
        )
        return (
            "CONTROLLED PLAYER NARRATIVE DIRECTION:\n"
            f"Current narrative direction: {objective}\n"
            "Treat this as an intention, not a mandatory milestone "
            "or a checklist to complete this turn.\n"
            "Respond naturally to the NPC, the atmosphere, "
            "relationships, uncertainty, and consequences.\n"
            "You may investigate, converse, negotiate, reflect, "
            "change tactics, encounter setbacks, or develop "
            "the relationship when appropriate.\n"
            "Keep the story meaningful without rushing toward "
            "a goal or deliberately stalling.\n"
            "Do not invent documents, evidence, NPC cooperation, "
            "or another character's actions and decisions.\n"
            "Do not claim objective completion yourself.\n"
            "Avoid mechanical checklist dialogue."
        )

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
        "Use the actual NPC response and established facts to choose "
        "one concrete next action toward the current goal.\n"
        "If you already have enough information to ask, inspect, "
        "apply, verify, or act, do so now rather than repeatedly "
        "discussing preparation.\n"
        "Do not repeatedly ask for information already established.\n"
        "Do not leave a productive interaction or end the conversation "
        "merely out of politeness while the current goal is unresolved.\n"
        "If genuinely blocked, investigate, negotiate, or try a "
        "different plausible approach.\n"
        "Never assume NPC cooperation, invent documents, fabricate "
        "evidence, or narrate another character's decisions.\n"
        "Do not independently mark goals as completed.\n"
        "Write natural roleplay; avoid mechanical checklist dialogue."
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
    messages, state, knowledge=None, criteria_progress=None,
    definition=None, mode="action",
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
        state, criteria_progress, definition, mode
    )

    if knowledge is not None:
        guidance += "\n\n" + format_player_knowledge(knowledge)

    result = [dict(message) for message in messages]
    result[0]["content"] += "\n\n" + guidance
    return result
