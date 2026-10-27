"""Auditable, resumable opportunity screening decisions."""

from hashlib import sha256
import json

EVENT_TYPE = "quest_opportunity_v1"


def _digest(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def make_opportunity_event(
    turn, observations, candidates, selected, evaluator_id,
    metadata=None,
):
    if type(turn) is not int or turn < 1:
        raise ValueError("Invalid opportunity turn")

    player = [
        item["text"] for item in observations
        if item["turn"] == turn and item["role"] == "player"
    ]
    npc = [
        item["text"] for item in observations
        if item["turn"] == turn and item["role"] == "npc"
    ]
    if len(player) != 1 or len(npc) != 1:
        raise ValueError("Missing or duplicate turn observations")

    allowed = {
        (entry["goal_id"], criterion_id)
        for entry in candidates
        for criterion_id in entry["criterion_ids"]
    }

    seen = set()
    for entry in selected:
        key = (entry["goal_id"], entry["criterion_id"])
        if key not in allowed or key in seen:
            raise ValueError("Invalid opportunity selection")
        seen.add(key)

    return {
        "type": EVENT_TYPE,
        "turn": turn,
        "evaluator_id": evaluator_id,
        "player_sha256": _digest(player[0]),
        "npc_sha256": _digest(npc[0]),
        "candidates_sha256": _digest(candidates),
        "selected": selected,
        "metadata": metadata,
    }


def read_opportunity_event(
    events, turn, observations, candidates, evaluator_id,
):
    matches = [
        event for event in events
        if event.get("type") == EVENT_TYPE
        and event.get("turn") == turn
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Duplicate opportunity event")

    event = matches[0]
    expected = make_opportunity_event(
        turn, observations, candidates, event["selected"],
        evaluator_id, event.get("metadata"),
    )
    if event != expected:
        raise ValueError("Opportunity event provenance mismatch")

    return event
