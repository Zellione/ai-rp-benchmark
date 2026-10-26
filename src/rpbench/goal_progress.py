"""Deterministic approval of semantic goal proposals.

The evaluator proposes. Explicit benchmark rules authorize.
Archived evaluations are replayed in chronological order.
"""

from copy import deepcopy

from .state import apply_confirmed_event


def validate_goal_rules(rules, state):
    if not isinstance(rules, list):
        raise ValueError("Goal approval rules must be an array")

    goals = state.get("goals")
    if not isinstance(goals, dict):
        raise ValueError("Invalid goal state")

    seen = set()

    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {
            "goal_id", "evidence_contains"
        }:
            raise ValueError("Invalid goal approval rule")

        goal_id = rule["goal_id"]
        phrase = rule["evidence_contains"]

        if (
            not isinstance(goal_id, str)
            or goal_id not in goals
            or goal_id in seen
        ):
            raise ValueError("Unknown or duplicate approval goal")

        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Invalid approval evidence")

        seen.add(goal_id)


def replay_goal_progress(initial, evaluations, rules):
    """Reconstruct approved goals without changing source data.

    Each accepted transition is applied through the existing
    verified state-event mechanism.
    """
    validate_goal_rules(rules, initial)

    state = deepcopy(initial)
    policies = {
        rule["goal_id"]: rule["evidence_contains"].casefold()
        for rule in rules
    }

    approved = []

    for turn in sorted(evaluations):
        event = evaluations[turn]

        if event.get("turn") != turn:
            raise ValueError("Evaluation turn mismatch")

        for proposal in event["result"]["goal_updates"]:
            goal_id = proposal["goal_id"]

            if goal_id not in policies:
                continue

            if proposal["status"] != "completed":
                continue

            goal = state["goals"][goal_id]
            if goal["status"] in {"completed", "failed"}:
                continue

            evidence = proposal["evidence"]

            if policies[goal_id] not in evidence.casefold():
                continue

            transition = {
                "type": "goal_status_changed",
                "event_id": state["last_confirmed_event"] + 1,
                "verified": True,
                "verified_by": "rule",
                "goal_id": goal_id,
                "status": "completed",
            }

            state = apply_confirmed_event(state, transition)

            approved.append({
                "turn": turn,
                "goal_id": goal_id,
                "status": "completed",
                "evidence": evidence,
                "authority": "configured_rule",
            })

    return state, approved



def refresh_goal_progress(root, observations, rules):
    """Reconstruct approved state from immutable ground truth and journal."""
    from .core import dump, load
    from .evaluation_journal import read_evaluations
    from .state import initial_state

    initial = initial_state(load(root / "ground_truth.json"))
    validate_goal_rules(rules, initial)

    evaluations = read_evaluations(
        root / "events.jsonl", observations, initial
    )
    state, approvals = replay_goal_progress(
        initial, evaluations, rules
    )
    dump(root / "state.json", state)
    return state, approvals
