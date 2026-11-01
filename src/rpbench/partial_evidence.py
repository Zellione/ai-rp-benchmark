"""Retrieve archived partial evidence for cumulative goal evaluation."""


def prior_partial_evidence(
    criterion_progress, observations, current_turn, limit=3,
):
    """Return recent, distinct, verifiable earlier partial sources.

    The journal/replay owns evidence validation. We only use
    NPC observations strictly preceding the evaluated turn.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Invalid partial evidence limit")

    npc_by_turn = {
        item["turn"]: item["text"]
        for item in observations
        if item["role"] == "npc"
        and item["turn"] < current_turn
    }

    selected = {}

    for evidence in criterion_progress.get("evidence", []):
        if evidence.get("status") != "partial":
            continue

        turn = evidence.get("turn")
        reason = evidence.get("reason")

        if (
            type(turn) is not int
            or turn not in npc_by_turn
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            continue

        selected[turn] = {
            "turn": turn,
            "prior_assessment": reason,
            "npc_response": npc_by_turn[turn],
        }

    turns = sorted(selected)

    if len(turns) <= limit:
        retained = turns
    elif limit == 1:
        retained = turns[:1]
    else:
        # Keep the earliest partial evidence as an anchor.
        # Fill remaining slots with the most recent sources.
        retained = [turns[0]] + turns[-(limit - 1):]

    return [selected[turn] for turn in retained]
