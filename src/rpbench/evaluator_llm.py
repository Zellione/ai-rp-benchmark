"""Semantic evaluator using an independent chat-completion model."""

import json
import os

from .core import Backend
from .player_evaluator import validate_evaluation
from .sampling import validate_sampling


EVALUATOR_INSTRUCTIONS = """You evaluate a fictional roleplay conversation.

Extract only useful NEW information explicitly supported by the
latest NPC response.

Return one JSON object with exactly two arrays:

{
  "knowledge_updates": [
    {"fact": "Concise factual claim", "evidence": "Exact NPC quote"}
  ],
  "goal_updates": [
    {
      "goal_id": "Existing goal identifier",
      "status": "completed",
      "evidence": "Exact NPC quote"
    }
  ]
}

Rules:
- Evidence must be an exact substring of the NPC response.
- NPC claims are reported information, not verified world truth.
- Do not repeat information the player already knows.
- Never infer success from the player's intentions alone.
- Propose a completed goal only when its full meaning is satisfied.
- A mere name mention is not sufficient for a broad objective.
- Use empty arrays when no meaningful update is supported.
- Return JSON only, without Markdown.
"""


def evaluate_llm(npc_text, state, knowledge, settings):
    """Generate and validate a semantic evaluation proposal."""
    backend_cfg = settings["backend"]
    sampling = settings["sampling"]

    backend = Backend(
        name=backend_cfg["name"],
        base_url=backend_cfg["base_url"],
        model=backend_cfg["model"],
        mode="chat",
        api_key=os.getenv(
            backend_cfg.get("api_key_env", "RPBENCH_EVALUATOR_API_KEY")
        ),
        timeout=backend_cfg.get("timeout", 300),
    )

    validate_sampling(backend.name, backend.mode, sampling)

    context = {
        "goals": state["goals"],
        "known_information": [
            {
                "fact": item["fact"],
                "status": item["status"],
            }
            for item in knowledge["facts"]
        ],
        "latest_npc_response": npc_text,
    }

    messages = [
        {
            "role": "system",
            "content": EVALUATOR_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": json.dumps(
                context, ensure_ascii=False
            ),
        },
    ]

    url, payload = backend.prepare(messages, sampling, {})
    text, _ = backend.generate(url, payload)

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Evaluator returned invalid JSON"
        ) from exc

    # The evaluation journal performs the authoritative
    # validation and normalization before persisting.
    # Return the raw proposal to avoid double validation.
    return result
