"""Player-observable narrative segmentation."""

import json
import os
import re

from .core import Backend
from .evaluator_llm import parse_evaluator_json


GATE_INSTRUCTIONS = 'You identify PLAYER-PERSPECTIVE SCENE BOUNDARIES\nin a fictional roleplay narrative.\n\nThe player action establishes the player\'s location\nand current activity.\n\nRead the numbered paragraphs IN ORDER.\nTrack where the player is throughout the scene.\n\nFor each paragraph, choose:\n- observable: Its information is directly accessible\n  to the player through sight, hearing, reading,\n  or an established communication channel.\n- offscreen: Its information is not accessible\n  to the player.\n- mixed: It contains both observable and offscreen\n  information that cannot be safely separated\n  at paragraph boundaries.\n\nThe narrator\'s knowledge is not the player\'s knowledge.\nA private NPC conversation elsewhere is offscreen.\nPhone conversations with the player are observable.\nDo not assume the player can hear distant speech.\n\nIMPORTANT:\n- Track scene continuity between paragraphs.\n- Once the player leaves, later conversation at the\n  former location remains offscreen unless the\n  narrative explicitly establishes a new channel.\n- Do not infer that an NPC\'s opinion is true.\n- Do not extract facts or evaluate goals.\n- Classify accessibility only.\n- When uncertain, choose offscreen.\n\nReturn exactly one JSON object:\n{\n  "labels": [\n    {"id": 0, "visibility": "observable"}\n  ]\n}\n\nReturn one label per supplied paragraph in order.\nUse only observable, offscreen, or mixed.\nNo explanations. Return JSON only.\n'


def split_paragraphs(text):
    if not isinstance(text, str):
        raise ValueError("NPC narrative must be text")

    return [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]


def validate_gate_result(result, count):
    if not isinstance(result, dict) or set(result) != {"labels"}:
        raise ValueError("Invalid scene boundary schema")

    labels = result["labels"]

    if not isinstance(labels, list) or len(labels) != count:
        raise ValueError("Wrong number of scene labels")

    allowed = {"observable", "offscreen", "mixed"}

    for index, entry in enumerate(labels):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "visibility"}
            or type(entry["id"]) is not int
            or entry["id"] != index
            or entry["visibility"] not in allowed
        ):
            raise ValueError(f"Invalid scene label at {index}")

    return [entry["visibility"] for entry in labels]


def observable_text(paragraphs, labels):
    if len(paragraphs) != len(labels):
        raise ValueError("Paragraph and label count mismatch")

    return "\n\n".join(
        paragraph
        for paragraph, label in zip(paragraphs, labels)
        if label == "observable"
    )


def evaluate_gate(
    player_action, npc_text, settings,
    on_response=None, max_tokens=4096,
):
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError("Invalid gate token budget")

    parts = split_paragraphs(npc_text)

    if not parts:
        return [], {
            "content": "",
            "reasoning": "",
            "finish_reason": "empty",
            "usage": None,
        }

    cfg = settings["backend"]
    backend = Backend(
        name=cfg["name"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        mode="chat",
        api_key=os.getenv(
            cfg.get("api_key_env", "RPBENCH_EVALUATOR_API_KEY")
        ),
        timeout=cfg.get("timeout", 300),
    )

    messages = [
        {"role": "system", "content": GATE_INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps({
                "player_action": player_action,
                "paragraphs": [
                    {"id": i, "text": part}
                    for i, part in enumerate(parts)
                ],
            }, ensure_ascii=False),
        },
    ]

    sampling = dict(settings["sampling"])
    sampling["max_tokens"] = max_tokens

    url, payload = backend.prepare(messages, sampling, {})
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

    if metadata["finish_reason"] == "length":
        raise ValueError("Knowledge gate exhausted token budget")

    labels = validate_gate_result(
        parse_evaluator_json(text), len(parts)
    )

    return labels, metadata
