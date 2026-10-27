"""Conservative normalization of knowledge-only LLM proposals."""


def normalize_knowledge_proposal(proposal, visible_text, turn):
    """Repair narrow schema deviations without inventing evidence."""
    if not isinstance(proposal, dict):
        return proposal

    if set(proposal) != {"knowledge_updates", "goal_updates"}:
        return proposal

    updates = proposal["knowledge_updates"]
    if not isinstance(updates, list):
        return proposal

    normalized = []

    for item in updates:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        candidate = dict(item)

        # Singular field name, but already an array.
        if (
            "source_turn" in candidate
            and "source_turns" not in candidate
            and set(candidate) <= {
                "fact", "source_turn", "evidence"
            }
            and candidate["source_turn"] == [turn]
        ):
            candidate["source_turns"] = candidate.pop(
                "source_turn"
            )

        if (
            set(candidate) == {
                "fact", "source_turns", "evidence"
            }
            and candidate["source_turns"] == [turn]
            and isinstance(candidate["evidence"], str)
            and candidate["evidence"].strip()
            and candidate["evidence"] in visible_text
        ):
            candidate.pop("evidence")

        normalized.append(candidate)

    return {
        "knowledge_updates": normalized,
        "goal_updates": proposal["goal_updates"],
    }
