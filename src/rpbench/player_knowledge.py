"""Deterministic, evidence-backed player knowledge tracking.

Reported claims are NOT treated as verified world facts.
"""

from copy import deepcopy


def validate_knowledge_rules(rules):
    if not isinstance(rules, list):
        raise ValueError("Knowledge rules must be a list")

    seen = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Invalid knowledge rule")

        if set(rule) != {"id", "npc_contains", "fact"}:
            raise ValueError("Unexpected knowledge rule fields")

        for key in ("id", "npc_contains", "fact"):
            if (
                not isinstance(rule[key], str)
                or not rule[key].strip()
            ):
                raise ValueError(f"Invalid knowledge rule {key}")

        if rule["id"] in seen:
            raise ValueError("Duplicate knowledge rule ID")

        seen.add(rule["id"])


def initial_player_knowledge(player):
    facts = player.get("known_facts", [])

    if (
        not isinstance(facts, list)
        or any(
            not isinstance(fact, str) or not fact.strip()
            for fact in facts
        )
    ):
        raise ValueError("Invalid initial player knowledge")

    return {
        "schema_version": 1,
        "facts": [
            {
                "id": f"initial-{index}",
                "fact": fact,
                "status": "known",
                "source": "player_profile",
                "turn": 0,
            }
            for index, fact in enumerate(facts, 1)
        ],
    }


def update_player_knowledge(knowledge, turn, npc_text, rules):
    """Record rule-matched NPC claims without assuming truth."""
    validate_knowledge_rules(rules)

    if type(turn) is not int or turn < 1:
        raise ValueError("Invalid knowledge turn")

    if not isinstance(npc_text, str):
        raise ValueError("Invalid NPC response")

    updated = deepcopy(knowledge)
    facts = updated["facts"]
    existing = {item["id"] for item in facts}

    for rule in rules:
        if rule["id"] in existing:
            continue

        needle = rule["npc_contains"]
        index = npc_text.casefold().find(needle.casefold())

        if index < 0:
            continue

        evidence = npc_text[index:index + len(needle)]

        facts.append({
            "id": rule["id"],
            "fact": rule["fact"],
            "status": "reported",
            "source": "npc_dialogue",
            "turn": turn,
            "evidence": evidence,
        })
        existing.add(rule["id"])

    return updated


def replay_player_knowledge(player, observations, rules):
    """Rebuild knowledge from archived NPC observations."""
    knowledge = initial_player_knowledge(player)

    for entry in observations:
        if entry["role"] == "npc":
            knowledge = update_player_knowledge(
                knowledge,
                entry["turn"],
                entry["text"],
                rules,
            )

    return knowledge


def format_player_knowledge(knowledge):
    lines = ["PLAYER KNOWLEDGE:"]

    facts = knowledge["facts"]

    if not facts:
        return (
            "PLAYER KNOWLEDGE:\n"
            "No tracked facts or claims acquired yet."
        )

    for entry in facts:
        status = entry["status"]
        fact = entry["fact"]

        if status == "known":
            lines.append(f"- [KNOWN] {fact}")
        elif status == "reported":
            lines.append(
                f"- [REPORTED, turn {entry['turn']}] {fact}"
            )
        else:
            raise ValueError("Unsupported knowledge status")

    lines.append(
        "Reported statements are claims, not verified facts. "
        "Do not assume their truth without further evidence."
    )

    return "\n".join(lines)
