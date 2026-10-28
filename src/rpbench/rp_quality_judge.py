"""Blind evaluation of complete adaptive RP conversations."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Backend
from .evaluator_llm import parse_evaluator_json


DIMENSIONS = {
    "german_prose": 25,
    "character_fidelity": 20,
    "dialogue_responsiveness": 20,
    "subtext": 15,
    "narrative_initiative": 10,
    "continuity": 10,
}

INSTRUCTIONS = """You are an independent evaluator of German
literary roleplay. Evaluate the NPC's performance across the
entire conversation.

The player messages provide context, but DO NOT score the
player's writing quality, personality, or initiative.

Assess these dimensions on a scale from 1 to 10:

german_prose: Natural, idiomatic, expressive German writing.
Penalize awkward grammar, clichés and repetitive imagery.

character_fidelity: Consistency with the character's personality,
motivations, knowledge boundaries and emotional restraint.

dialogue_responsiveness: Meaningful reactions to actual player
statements, questions, objections and actions.

subtext: Credible emotions, tension and implications that emerge
through behavior rather than excessive explanation.

narrative_initiative: NPC contributes meaningful developments
without forcing the player into predetermined outcomes.

continuity: Spatial, chronological, causal and interpersonal
consistency across all turns.

Important:
- Do not equate verbosity with quality.
- Do not reward melodrama merely because it sounds literary.
- Distinguish deliberate stylistic repetition from repetitive
  or formulaic output.
- Penalize unsupported technical claims if they undermine the
  credibility of a character's expertise.
- Evaluate player agency separately: NPC must not dictate the
  player's thoughts, feelings, speech or unchosen actions.
- Do not assume a hidden character secret should be disclosed.
  Evaluate whether the character handles it plausibly.
- Do not infer model identity from writing style.
- Do not rank candidates or compare to other conversations.
- Quote brief, exact NPC excerpts as evidence. Do not invent
  quotations or cite player text as NPC evidence.

Return ONLY one JSON object:
{
  "scores": {
    "german_prose": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    },
    "character_fidelity": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    },
    "dialogue_responsiveness": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    },
    "subtext": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    },
    "narrative_initiative": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    },
    "continuity": {
      "score": 1,
      "reason": "...",
      "evidence": ["exact NPC excerpt"]
    }
  },
  "player_agency_violations": [],
  "major_consistency_issues": [],
  "overall_assessment": "..."
}

Every score must be an integer from 1 to 10.
Each dimension requires at least one NPC quotation.
Be critical and calibrated. Do not give automatic high scores.
"""


def validate_judgment(proposal, transcript):
    if not isinstance(proposal, dict):
        raise ValueError("Judgment must be an object")

    if set(proposal) != {
        "scores", "player_agency_violations",
        "major_consistency_issues", "overall_assessment"
    }:
        raise ValueError("Invalid judgment fields")

    scores = proposal["scores"]
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError("Invalid dimension set")

    npc_texts = [
        item["text"] for item in transcript
        if item["role"] == "npc"
    ]

    for name, entry in scores.items():
        if not isinstance(entry, dict) or set(entry) != {
            "score", "reason", "evidence"
        }:
            raise ValueError(f"Invalid dimension entry: {name}")

        if type(entry["score"]) is not int or not 1 <= entry["score"] <= 10:
            raise ValueError(f"Invalid score: {name}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError(f"Missing reason: {name}")
        evidence = entry["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Missing evidence: {name}")
        for excerpt in evidence:
            if (
                not isinstance(excerpt, str)
                or not excerpt.strip()
                or not any(excerpt in text for text in npc_texts)
            ):
                raise ValueError(f"Unsupported NPC quotation: {name}")

    for field in ("player_agency_violations", "major_consistency_issues"):
        values = proposal[field]
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"Invalid {field}")

    if (
        not isinstance(proposal["overall_assessment"], str)
        or not proposal["overall_assessment"].strip()
    ):
        raise ValueError("Missing overall assessment")

    return round(sum(
        scores[key]["score"] * weight
        for key, weight in DIMENSIONS.items()
    ) / 100, 3)


def evaluate(case, backend, sampling, context):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": json.dumps({
                "character_and_scenario": context,
                "conversation": case["transcript"],
            }, ensure_ascii=False),
        },
    ]

    url, payload = backend.prepare(messages, sampling, {})
    start = time.perf_counter()
    captured = {}

    try:
        content, response = backend.generate(url, payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        captured = {
            "content": content,
            "reasoning": (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            ),
            "usage": response.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }

        if captured["finish_reason"] == "length":
            raise ValueError("Judge generation truncated")

        proposal = parse_evaluator_json(content)
        total = validate_judgment(proposal, case["transcript"])

        captured["valid"] = True
        captured["judgment"] = proposal
        captured["weighted_score"] = total

    except Exception as error:
        captured["valid"] = False
        captured["error"] = f"{type(error).__name__}: {error}"

    captured["elapsed_seconds"] = round(
        time.perf_counter() - start, 3
    )
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    if not 1 <= args.repetitions <= 10:
        parser.error("repetitions must be 1..10")

    dataset_bytes = Path(args.dataset).read_bytes()
    dataset = json.loads(dataset_bytes)
    if (
        dataset.get("schema_version") != 1
        or not isinstance(dataset.get("cases"), list)
        or not dataset["cases"]
    ):
        parser.error("Invalid evaluation dataset")

    scenario = json.loads(
        Path(args.scenario).read_text(encoding="utf-8")
    )

    backend = Backend(
        name="nanogpt",
        base_url="https://nano-gpt.com/api",
        model="z-ai/glm-5.3-flash",
        mode="chat",
        api_key=os.environ.get("NANOGPT_API_KEY"),
        timeout=900,
    )
    if not backend.api_key:
        parser.error("NANOGPT_API_KEY is not set")

    sampling = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 32768,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        for repetition in range(1, args.repetitions + 1):
            for case in dataset["cases"]:
                print(
                    f"Judging {case['id']} | repetition {repetition}",
                    flush=True,
                )
                record = evaluate(
                    case, backend, sampling, scenario["system"]
                )
                record.update({
                    "candidate": case["id"],
                    "repetition": repetition,
                    "judge_model": backend.model,
                    "sampling": sampling,
                    "dataset_sha256": hashlib.sha256(
                        dataset_bytes
                    ).hexdigest(),
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                file.flush()
                print(
                    f"  valid={record['valid']} "
                    f"score={record.get('weighted_score')} "
                    f"tokens={(record.get('usage') or {}).get('completion_tokens')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
