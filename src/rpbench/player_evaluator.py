"""Validate semantic player-knowledge and goal proposals.

Evaluator output is untrusted. Validation does not imply that
a claim is true or that a goal has actually been achieved.
"""

from collections.abc import Mapping


def validate_evaluation(result, npc_text, state, turn=None):
    if not isinstance(result, Mapping):
        raise ValueError("Evaluator result must be an object")

    if set(result) != {"knowledge_updates", "goal_updates"}:
        raise ValueError("Invalid evaluator result fields")

    if not isinstance(npc_text, str):
        raise ValueError("Invalid NPC evidence source")

    goals = state.get("goals") if isinstance(state, Mapping) else None
    if not isinstance(goals, Mapping):
        raise ValueError("Invalid goal state")

    knowledge = result["knowledge_updates"]
    updates = result["goal_updates"]

    if not isinstance(knowledge, list) or not isinstance(updates, list):
        raise ValueError("Evaluator updates must be arrays")

    if len(knowledge) > 20 or len(updates) > len(goals):
        raise ValueError("Too many evaluator updates")

    seen_goals = set()
    clean_knowledge = []
    clean_goals = []

    for item in knowledge:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid knowledge proposal")

        if set(item) == {"fact", "source_turns"}:
            fact = item["fact"]
            turns = item["source_turns"]

            if not isinstance(fact, str) or not fact.strip():
                raise ValueError("Empty knowledge fact")

            if (
                type(turn) is not int
                or not isinstance(turns, list)
                or turns != [turn]
            ):
                raise ValueError("Invalid knowledge source turn")

            clean_knowledge.append({
                "fact": fact.strip(),
                "source_turns": [turn],
                "status": "reported",
            })
            continue

        if set(item) != {"fact", "evidence"}:
            raise ValueError("Invalid knowledge proposal")

        fact = item["fact"]
        evidence = item["evidence"]

        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("Empty knowledge fact")

        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Missing knowledge evidence")

        if evidence not in npc_text:
            raise ValueError("Knowledge evidence not in NPC response")

        clean_knowledge.append({
            "fact": fact.strip(),
            "evidence": evidence,
            "status": "reported",
        })

    for item in updates:
        if not isinstance(item, Mapping) or set(item) != {
            "goal_id", "status", "evidence",
        }:
            raise ValueError("Invalid goal proposal")

        goal_id = item["goal_id"]
        status = item["status"]
        evidence = item["evidence"]

        if goal_id not in goals or goal_id in seen_goals:
            raise ValueError("Unknown or duplicate goal")

        if status not in {"completed", "failed"}:
            raise ValueError("Unsupported goal transition")

        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Missing goal evidence")

        if evidence not in npc_text:
            raise ValueError("Goal evidence not in NPC response")

        if goals[goal_id].get("status") in {"completed", "failed"}:
            raise ValueError("Goal already terminal")

        seen_goals.add(goal_id)
        clean_goals.append({
            "goal_id": goal_id,
            "status": status,
            "evidence": evidence,
        })

    return {
        "knowledge_updates": clean_knowledge,
        "goal_updates": clean_goals,
    }
