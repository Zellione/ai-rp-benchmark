"""Apply semantic criterion progress to the controlled player.

Model-supported completions are explicitly marked as adjudicator
decisions. Source provenance is validated, but semantic correctness
is not independently guaranteed.
"""

from copy import deepcopy

from .semantic_criteria import replay_semantic_criteria
from .state import apply_confirmed_event


def advance_criteria_goals(
    initial_state,
    definition,
    observations,
    semantic_events,
):
    """Replay criterion decisions and advance satisfied goals.

    Never mutate the input state or the source events.
    """
    progress = replay_semantic_criteria(
        definition,
        initial_state,
        observations,
        semantic_events,
    )

    updated = deepcopy(initial_state)
    transitions = []

    for goal_id, result in progress["goals"].items():
        if not result["proposed_satisfied"]:
            continue

        if updated["goals"][goal_id]["status"] in {
            "completed", "failed"
        }:
            continue

        transition = {
            "type": "goal_status_changed",
            "event_id": updated["last_confirmed_event"] + 1,
            "verified": True,
            "verified_by": "adjudicator",
            "goal_id": goal_id,
            "status": "completed",
        }

        updated = apply_confirmed_event(
            updated, transition
        )

        transitions.append({
            "goal_id": goal_id,
            "status": "completed",
            "authority": "semantic_model_adjudicator",
            "evidence": {
                criterion_id: item["evidence"]
                for criterion_id, item in result[
                    "criteria"
                ].items()
            },
        })

    return updated, progress, transitions
