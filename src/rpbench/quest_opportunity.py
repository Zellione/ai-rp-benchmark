"""Small semantic prefilter for unexpected quest progress."""

import json
import os

from .core import Backend
from .evaluator_llm import parse_evaluator_json


INSTRUCTIONS = """You detect POSSIBLE early quest successes.

You are not a goal adjudicator. Do not decide completion.

Read the latest player action and NPC response.
Identify candidate criteria that might have been fulfilled
by these events, even if this was not the player's intent.

Be recall-oriented: include plausible successes when uncertain.
Do not report criteria with no concrete supporting event.

Do not confuse:
- requesting authorization with obtaining authorization
- obtaining authorization with examining a document
- accessing a public document with a restricted document
- intentions and promises with completed actions

Return exactly:
{"possible": [{"goal_id": "...", "criterion_id": "..."}]}

Use only supplied candidate IDs.
Return {"possible": []} if there are no plausible matches.
No Markdown or explanations.
"""


def parse_opportunities(text, candidates):
    result = parse_evaluator_json(text)

    if not isinstance(result, dict) or set(result) != {"possible"}:
        raise ValueError("Invalid opportunity response")

    items = result["possible"]
    if not isinstance(items, list):
        raise ValueError("Invalid opportunity list")

    allowed = {
        (item["goal_id"], criterion_id)
        for item in candidates
        for criterion_id in item["criterion_ids"]
    }

    selected = []
    seen = set()

    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "goal_id", "criterion_id"
        }:
            raise ValueError("Invalid opportunity entry")

        key = (item["goal_id"], item["criterion_id"])

        if key not in allowed:
            raise ValueError("Unknown opportunity target")
        if key in seen:
            raise ValueError("Duplicate opportunity")

        seen.add(key)
        selected.append({
            "goal_id": key[0],
            "criterion_id": key[1],
        })

    return selected


def check_opportunities(
    definition, candidates, observations, settings,
):
    """Return potential early-success targets; no state mutation."""
    if not candidates:
        return [], None

    latest = next(
        (
            item for item in reversed(observations)
            if item["role"] == "npc"
        ),
        None,
    )
    if latest is None:
        raise ValueError("Missing NPC response")

    player_action = next(
        (
            item["text"]
            for item in reversed(observations)
            if item["role"] == "player"
            and item["turn"] == latest["turn"]
        ),
        None,
    )

    if player_action is None:
        raise ValueError("Missing same-turn player action")

    descriptions = []

    for candidate in candidates:
        goal_id = candidate["goal_id"]
        requested = set(candidate["criterion_ids"])

        for criterion in definition["goals"][goal_id]["criteria"]:
            if criterion["id"] in requested:
                descriptions.append({
                    "goal_id": goal_id,
                    "criterion_id": criterion["id"],
                    "description": criterion["description"],
                })

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

    context = {
        "candidate_criteria": descriptions,
        "latest_player_action": player_action,
        "latest_npc_response": latest["text"],
    }

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False),
        },
    ]

    sampling = {
        **settings["sampling"],
        "max_tokens": min(
            settings["sampling"].get("max_tokens", 4096),
            2048,
        ),
    }

    url, payload = backend.prepare(messages, sampling, {})
    content, response = backend.generate(url, payload)

    choice = (response.get("choices") or [{}])[0]

    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated opportunity response")

    selected = parse_opportunities(content, candidates)

    return selected, {
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }
