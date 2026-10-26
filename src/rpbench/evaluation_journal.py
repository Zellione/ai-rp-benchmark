"""Auditable semantic evaluation records for RPBench."""

from hashlib import sha256
import json
from pathlib import Path

from .core import append
from .observations import read_jsonl
from .player_evaluator import validate_evaluation


EVENT_TYPE = "player_evaluation_v1"


def response_hash(text):
    if not isinstance(text, str):
        raise ValueError("Invalid NPC response")
    return sha256(text.encode("utf-8")).hexdigest()


def make_evaluation_event(
    turn, npc_text, state, proposal, evaluator_id
):
    """Validate a proposal and bind it to its source response."""
    if type(turn) is not int or turn < 1:
        raise ValueError("Invalid evaluation turn")

    if (
        not isinstance(evaluator_id, str)
        or not evaluator_id.strip()
    ):
        raise ValueError("Missing evaluator ID")

    validated = validate_evaluation(
        proposal, npc_text, state, turn=turn
    )

    return {
        "type": EVENT_TYPE,
        "turn": turn,
        "npc_response_sha256": response_hash(npc_text),
        "evaluator_id": evaluator_id,
        "result": validated,
    }


def read_evaluations(events_path, observations, initial_state):
    """Validate archived evaluations against their NPC responses.

    No archived file is modified. Duplicate/conflicting
    evaluations are rejected rather than silently replaced.
    """
    path = Path(events_path)
    events = read_jsonl(path) if path.exists() else []

    responses = {}
    for observation in observations:
        if observation["role"] != "npc":
            continue

        turn = observation["turn"]
        if turn in responses:
            raise ValueError("Duplicate NPC observation")
        responses[turn] = observation["text"]

    evaluations = {}

    for event in events:
        if event.get("type") != EVENT_TYPE:
            continue

        turn = event.get("turn")
        if type(turn) is not int or turn not in responses:
            raise ValueError("Evaluation lacks NPC observation")

        if turn in evaluations:
            raise ValueError("Duplicate evaluation for turn")

        if set(event) != {
            "type", "turn", "npc_response_sha256",
            "evaluator_id", "result",
        }:
            raise ValueError("Invalid evaluation event fields")

        if event["npc_response_sha256"] != response_hash(
            responses[turn]
        ):
            raise ValueError("Evaluation response hash mismatch")

        # Archived evaluations contain the normalized status
        # added by validate_evaluation(). Convert back to the
        # raw proposal shape before validating provenance.
        archived_result = event["result"]

        if (
            not isinstance(archived_result, dict)
            or set(archived_result) != {
                "knowledge_updates", "goal_updates"
            }
            or not isinstance(
                archived_result["knowledge_updates"], list
            )
        ):
            raise ValueError("Invalid archived evaluation result")

        raw_knowledge = []

        for item in archived_result["knowledge_updates"]:
            if (
                not isinstance(item, dict)
                or item.get("status") != "reported"
            ):
                raise ValueError("Invalid archived knowledge proposal")

            if set(item) == {"fact", "source_turns", "status"}:
                raw_knowledge.append({
                    "fact": item["fact"],
                    "source_turns": item["source_turns"],
                })
            elif set(item) == {"fact", "evidence", "status"}:
                raw_knowledge.append({
                    "fact": item["fact"],
                    "evidence": item["evidence"],
                })
            else:
                raise ValueError("Invalid archived knowledge fields")

        raw_proposal = {
            "knowledge_updates": raw_knowledge,
            "goal_updates": archived_result["goal_updates"],
        }

        verified = make_evaluation_event(
            turn,
            responses[turn],
            initial_state,
            raw_proposal,
            event["evaluator_id"],
        )

        if verified != event:
            raise ValueError("Evaluation record mismatch")

        evaluations[turn] = event

    return evaluations


def append_evaluation(
    events_path,
    turn,
    npc_text,
    state,
    proposal,
    evaluator_id,
    observations,
):
    """Append a unique evaluation after verifying provenance."""
    event = make_evaluation_event(
        turn, npc_text, state, proposal, evaluator_id
    )

    known = read_evaluations(
        events_path, observations, state
    )

    if turn in known:
        raise ValueError("Evaluation already archived")

    matches = [
        item for item in observations
        if item["role"] == "npc"
        and item["turn"] == turn
    ]

    if (
        len(matches) != 1
        or matches[0]["text"] != npc_text
    ):
        raise ValueError("NPC observation mismatch")

    append(Path(events_path), event)
    return event
