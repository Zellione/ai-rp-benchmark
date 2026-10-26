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
- Every evidence field MUST be one exact, contiguous substring
  copied character-for-character from the latest NPC response.
- Never abbreviate quotations, insert ellipses, join separate
  passages, normalize punctuation, or paraphrase evidence.
- If several passages are needed, make separate knowledge updates.
- A goal completion must be supported by one sufficient quote.
  Otherwise do not propose completion.
- NPC claims are reported information, not verified world truth.
- Do not repeat information the player already knows.
- Never infer success from the player's intentions alone.
- Propose a completed goal only when its full meaning is satisfied.
- Partial progress is not completion. Knowing that written
  authorization is required does not necessarily establish who
  grants it or the complete authorization procedure.
- Do not complete broad goals based on partial staff information.
- A mere name mention is not sufficient for a broad objective.
- Use empty arrays when no meaningful update is supported.
- Return JSON only, without Markdown.
"""



def parse_evaluator_json(text):
    """Accept plain JSON or one complete fenced JSON block."""
    if not isinstance(text, str):
        raise ValueError("Evaluator content must be text")

    content = text.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```", "```json"}
            and lines[-1].strip() == "```"
        ):
            content = "\n".join(lines[1:-1])
        # Any malformed fence remains intact and will be
        # rejected by json.loads().

    return json.loads(content)


def evaluate_llm(
    npc_text, state, knowledge, settings,
    on_response=None, knowledge_only=False,
):
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

    instructions = EVALUATOR_INSTRUCTIONS
    if knowledge_only:
        instructions += (
            "\nKNOWLEDGE-ONLY MODE:\n"
            "Extract information, but do NOT evaluate goal completion.\n"
            "Always return an empty goal_updates array.\n"
        )

    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": json.dumps(
                context, ensure_ascii=False
            ),
        },
    ]

    url, payload = backend.prepare(messages, sampling, {})
    text, response = backend.generate(url, payload)

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    usage = response.get("usage")
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )

    if on_response is not None:
        on_response({
            "content": text,
            "reasoning": reasoning,
            "finish_reason": finish_reason,
            "usage": usage,
        })

    try:
        result = parse_evaluator_json(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Evaluator returned invalid JSON; "
            f"finish_reason={finish_reason!r}; "
            f"visible_chars={len(text)}; "
            f"reasoning_chars={len(reasoning)}; "
            f"usage={usage!r}; "
            f"content_preview={text[:300]!r}"
        ) from exc

    if knowledge_only and isinstance(result, dict):
        # The dedicated criteria evaluator owns all goal decisions.
        # Only the knowledge portion enters the legacy journal.
        result["goal_updates"] = []

    # The evaluation journal performs the authoritative
    # validation and normalization before persisting.
    # Return the raw proposal to avoid double validation.
    return result
