"""Deterministic planning for economical goal adjudication."""

from .player_controller import select_player_goal
from .goal_criteria import prerequisites_met


def plan_goal_evaluations(state, definition, progress):
    """Select the normal goal and possible early-success targets.

    This function performs no model calls and changes no state.
    """
    current = select_player_goal(state, definition)

    active_goal_id = (
        current["id"]
        if current is not None
        and current["id"] in definition["goals"]
        else None
    )

    candidates = []

    for goal_id, entry in definition["goals"].items():
        status = state["goals"][goal_id]["status"]

        if not prerequisites_met(definition, state, goal_id):
            continue

        if status in {"completed", "skipped", "failed"}:
            continue

        pending = [
            criterion["id"]
            for criterion in entry["criteria"]
            if not progress["goals"][goal_id]["criteria"][
                criterion["id"]
            ]["proposed_satisfied"]
        ]

        if not pending:
            continue

        if goal_id == active_goal_id:
            continue

        candidates.append({
            "goal_id": goal_id,
            "criterion_ids": pending,
        })

    return {
        "active_goal_id": active_goal_id,
        "opportunity_candidates": candidates,
    }
