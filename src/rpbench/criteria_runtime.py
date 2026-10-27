"""Optional live semantic criteria evaluation and goal progression."""

import json
import os
import socket
import urllib.error
from pathlib import Path

from .core import Backend, append, dump, load
from .progress_status import emit
from .evaluator_llm import parse_evaluator_json
from .goal_criteria import validate_goal_criteria
from .semantic_criteria import (
    EVENT_TYPE,
    make_semantic_event,
)
from .criteria_progress import advance_criteria_goals
from .observations import read_jsonl
from .state import initial_state


INSTRUCTIONS = """You are an independent roleplay goal adjudicator.

Evaluate exactly ONE supplied goal and its open criteria.

Use the actual player action and NPC response. Semantic
paraphrases and unambiguous implications are acceptable,
but do not invent actions, permissions, documents, or facts.

Status definitions:

- satisfied: Every essential element of the exact criterion
  has been demonstrated by concrete evidence.

- partial: Concrete evidence demonstrates at least one
  identifiable, necessary component of THAT SAME criterion.
  The complete condition has not yet been fulfilled.

- not_satisfied: No required component of the exact criterion
  has been demonstrated, or evidence is insufficient.

Important distinctions:

- General preparation is not automatically partial progress.
- A related accomplishment is not a required component.
- Visitor registration is not a restricted-access application.
- Submitting an application is not receiving authorization.
- Public reading permission is not restricted authorization.
- Reading public records is not examining restricted records.
- An invitation to apply is not an approval.
- Hypothetical future actions are not completed actions.
- NPC helpfulness does not establish formal permission.

Judge each criterion independently. Match its exact scope.
Consider any exclusion conditions in its description.
Do not assume that a related goal was completed.
When essential evidence is absent, use not_satisfied.

Return exactly one JSON object:

{
  "schema_version": 1,
  "criterion_updates": [
    {
      "goal_id": "existing goal ID",
      "criterion_id": "existing criterion ID",
      "status": "not_satisfied",
      "source_turns": [],
      "reason": "Which required element was or was not demonstrated"
    }
  ]
}

Rules:
- Evaluate only supplied criteria.
- Return one update for every supplied criterion.
- Only cite existing NPC turns, numbered starting at 1.
- For satisfied and partial, provide at least one valid source turn.
- For not_satisfied, an empty source_turns list is valid.
- Never invent sources, goal IDs, or criterion IDs.
- NPC claims are not necessarily objective world truth.
- Negative judgments are valid results.
- Return only JSON, without Markdown.
"""


def _generate(
    definition, progress, observations, settings,
    on_response=None, goal_id=None,
):
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

    targets = (
        [goal_id]
        if goal_id is not None
        else list(definition["goals"])
    )

    for identifier in targets:
        if identifier not in definition["goals"]:
            raise ValueError("Unknown evaluation goal")

        pending = [
            {
                "id": criterion["id"],
                "description": criterion["description"],
            }
            for criterion in definition["goals"][identifier]["criteria"]
            if not progress["goals"][identifier]["criteria"][
                criterion["id"]
            ]["proposed_satisfied"]
        ]
        if pending:
            open_criteria[identifier] = pending

    # The player action matters for applications, permissions
    # and actual document examination.
    player_action = next(
        (
            item["text"]
            for item in reversed(observations)
            if item["role"] == "player"
            and item["turn"] == current["turn"]
        ),
        None,
    )

    context = {
        "active_goal_id": goal_id,
        "open_criteria": open_criteria,
        "latest_player_action": player_action,
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
    """Evaluate each (turn, goal) independently and resume safely."""
    root = Path(root)
    original = initial_state(load(root / "ground_truth.json"))
    definition = validate_goal_criteria(definition, original)

    events_path = root / "events.jsonl"
    events = read_jsonl(events_path) if events_path.exists() else []
    accepted = [
        event for event in events
        if event.get("type") == EVENT_TYPE
    ]

    _, progress, _ = advance_criteria_goals(
        original, definition, observations, accepted
    )

    checked = set()
    for event in accepted:
        if "goal_id" in event:
            checked.add((event["turn"], event["goal_id"]))
        else:
            for identifier in definition["goals"]:
                checked.add((event["turn"], identifier))

    for observation in observations:
        if observation["role"] != "npc":
            continue

        turn = observation["turn"]
        available = [
            entry for entry in observations
            if entry["turn"] <= turn
        ]

        for goal_id in definition["goals"]:
            if (turn, goal_id) in checked:
                continue

            # A fully satisfied goal need not be checked again.
            if progress["goals"][goal_id]["proposed_satisfied"]:
                continue

            if all(
                item["proposed_satisfied"]
                for item in progress["goals"][goal_id][
                    "criteria"
                ].values()
            ):
                continue

            max_attempts = settings.get("max_attempts", 1)

            for attempt in range(1, max_attempts + 1):
                emit(
                    f"Turn {turn:03} | CRITERIA {goal_id}"
                    f" (attempt {attempt}/{max_attempts})"
                )
                captured = {}

                def capture(response):
                    captured.update(response)

                try:
                    proposal, metadata = _generate(
                        definition, progress, available, settings,
                        on_response=capture,
                        goal_id=goal_id,
                    )

                    if not isinstance(proposal, dict):
                        raise ValueError("Invalid criteria result")

                    for update in proposal.get(
                        "criterion_updates", []
                    ):
                        for reference in update.get(
                            "evidence", []
                        ):
                            if reference.get("turn") != turn:
                                raise ValueError(
                                    "New evidence must cite current turn"
                                )

                    event = make_semantic_event(
                        turn, proposal, definition, original,
                        available, settings["id"],
                        goal_id=goal_id,
                    )

                except (
                    ValueError,
                    TimeoutError,
                    socket.timeout,
                    urllib.error.URLError,
                ) as error:
                    append(events_path, {
                        "type": "criteria_attempt",
                        "turn": turn,
                        "goal_id": goal_id,
                        "attempt": attempt,
                        "status": "rejected",
                        "error": str(error),
                        "raw_response": captured or None,
                    })
                    emit(
                        f"Turn {turn:03} | CRITERIA {goal_id} "
                        f"REJECTED ({attempt}/{max_attempts}): "
                        f"{error}"
                    )
                    if attempt == max_attempts:
                        raise
                else:
                    append(events_path, event)
                    append(events_path, {
                        "type": "criteria_attempt",
                        "turn": turn,
                        "goal_id": goal_id,
                        "attempt": attempt,
                        "status": "accepted",
                        "usage": metadata["usage"],
                        "finish_reason": metadata["finish_reason"],
                    })

                    for update in event["result"]["criterion_updates"]:
                        emit(
                            f"Turn {turn:03} | "
                            f"{update['goal_id']}/"
                            f"{update['criterion_id']} = "
                            f"{update['status']}"
                        )

                    accepted.append(event)
                    checked.add((turn, goal_id))
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
