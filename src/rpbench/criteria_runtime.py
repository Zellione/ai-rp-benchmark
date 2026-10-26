"""Optional live semantic criteria evaluation and goal progression."""

import json
import os
from pathlib import Path

from .core import Backend, append, dump, load
from .evaluator_llm import parse_evaluator_json
from .goal_criteria import validate_goal_criteria
from .semantic_criteria import (
    EVENT_TYPE,
    make_semantic_event,
)
from .criteria_progress import advance_criteria_goals
from .observations import read_jsonl
from .state import initial_state


INSTRUCTIONS = """Evaluate goal criteria in a fictional roleplay.

Return exactly this JSON structure:
{
  "schema_version": 1,
  "criterion_updates": [
    {
      "goal_id": "existing goal id",
      "criterion_id": "existing criterion id",
      "status": "satisfied",
      "evidence": [{"turn": 1, "quote": "exact NPC substring"}]
    }
  ]
}

Rules:
- Only report criteria fully supported by dialogue evidence.
- Quotes must be exact contiguous substrings of NPC messages.
- Cite the correct NPC turn number.
- For NEW updates, cite only the latest NPC turn.
- Previous evidence is already recorded in existing_progress.
- Never invent criterion IDs, dialogue, or evidence.
- Partial progress is not satisfaction.
- Do not repeat already satisfied criteria.
- If nothing new is satisfied, return an empty array.
- Return JSON only.
"""


def _generate(definition, progress, observations, settings, on_response=None):
    backend_cfg = settings["backend"]
    backend = Backend(
        name=backend_cfg["name"],
        base_url=backend_cfg["base_url"],
        model=backend_cfg["model"],
        mode="chat",
        api_key=os.getenv(
            backend_cfg.get(
                "api_key_env", "RPBENCH_EVALUATOR_API_KEY"
            )
        ),
        timeout=backend_cfg.get("timeout", 300),
    )

    # Only send the newest NPC reply. Earlier evidence already
    # survives in existing_progress and in the archived events.
    latest = [
        item for item in observations
        if item["role"] == "npc"
    ]
    if not latest:
        raise ValueError("No NPC response to evaluate")

    current = latest[-1]

    open_criteria = {}
    for goal_id, definition_entry in definition["goals"].items():
        pending = [
            {
                "id": criterion["id"],
                "description": criterion["description"],
            }
            for criterion in definition_entry["criteria"]
            if not progress["goals"][goal_id]["criteria"][
                criterion["id"]
            ]["proposed_satisfied"]
        ]
        if pending:
            open_criteria[goal_id] = pending

    context = {
        "open_criteria": open_criteria,
        "existing_progress": progress["goals"],
        "latest_npc_response": {
            "turn": current["turn"],
            "text": current["text"],
        },
    }
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False),
        },
    ]

    url, payload = backend.prepare(
        messages, settings["sampling"], {}
    )
    text, response = backend.generate(url, payload)

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    metadata = {
        "content": text,
        "reasoning": (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        ),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }

    if on_response is not None:
        on_response(metadata)

    try:
        return parse_evaluator_json(text), metadata
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Invalid criteria JSON; "
            f"finish_reason={metadata['finish_reason']!r}; "
            f"usage={metadata['usage']!r}; "
            f"visible_chars={len(text)}"
        ) from exc


def refresh_criteria(root, definition, observations, settings):
    """Evaluate missing NPC turns, then reconstruct goal state.

    The original goal state is always reconstructed from ground truth.
    Existing accepted evaluations are never regenerated.
    """
    root = Path(root)
    original = initial_state(load(root / "ground_truth.json"))
    definition = validate_goal_criteria(definition, original)

    events_path = root / "events.jsonl"
    events = read_jsonl(events_path) if events_path.exists() else []
    accepted = [
        event for event in events
        if event.get("type") == EVENT_TYPE
    ]

    # Validates IDs, sources, hashes, duplicate turns and replay.
    _, progress, _ = advance_criteria_goals(
        original, definition, observations, accepted
    )

    accepted_turns = {event["turn"] for event in accepted}

    for observation in observations:
        if observation["role"] != "npc":
            continue

        turn = observation["turn"]
        if turn in accepted_turns:
            continue

        available = [
            entry for entry in observations
            if entry["turn"] <= turn
        ]

        max_attempts = settings.get("max_attempts", 1)

        for attempt in range(1, max_attempts + 1):
            captured = {}

            def capture(response):
                captured.update(response)

            try:
                proposal, metadata = _generate(
                    definition, progress, available, settings,
                    on_response=capture,
                )
                for update in proposal.get("criterion_updates", []):
                    for reference in update.get("evidence", []):
                        if reference.get("turn") != turn:
                            raise ValueError(
                                "New criteria evidence must cite "
                                "the current NPC turn"
                            )

                event = make_semantic_event(
                    turn, proposal, definition, original,
                    available, settings["id"],
                )
            except ValueError as error:
                append(events_path, {
                    "type": "criteria_attempt",
                    "turn": turn,
                    "attempt": attempt,
                    "status": "rejected",
                    "error": str(error),
                    "raw_response": captured or None,
                })
                if attempt == max_attempts:
                    raise
            else:
                append(events_path, event)
                append(events_path, {
                    "type": "criteria_attempt",
                    "turn": turn,
                    "attempt": attempt,
                    "status": "accepted",
                    "usage": metadata["usage"],
                    "finish_reason": metadata["finish_reason"],
                })
                accepted.append(event)
                accepted_turns.add(turn)
                _, progress, _ = advance_criteria_goals(
                    original, definition, observations, accepted
                )
                break

    state, progress, transitions = advance_criteria_goals(
        original, definition, observations, accepted
    )

    dump(root / "state.json", state)
    dump(root / "criteria_progress.json", progress)

    return state, progress, transitions
