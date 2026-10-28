"""Automated, position-balanced pairwise RP evaluation."""

import argparse
import hashlib
import itertools
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Backend
from .evaluator_llm import parse_evaluator_json


INSTRUCTIONS = """You are an independent judge of literary
German roleplay quality.

Compare two complete conversations, X and Y.
Evaluate ONLY the NPC's writing and behavior.
The player text is context, not material to score.

Consider:
- Natural, polished, idiomatic German prose
- Character fidelity and credible emotional behavior
- Dialogue responsiveness to the actual player
- Subtext and psychological plausibility
- NPC initiative without stealing player agency
- Spatial, temporal, and causal continuity

Do not equate length, verbosity, melodrama or complexity
with quality. Penalize repeated formulaic language,
unsupported expertise, contradictions and NPC control
of player thoughts, actions or decisions.

Both conversations may develop differently.
Judge how well each NPC handles its own unfolding scene.
Do not reward a conversation merely for reaching a plot
revelation or for having a more cooperative player.

Choose X, Y or tie. If the evidence does not justify
a clear preference, choose tie.

Provide short, EXACT NPC quotations supporting the
comparison. Do not quote player text as NPC evidence.
Do not infer model identity. Do not refer to any earlier
scores or rankings. Return ONLY this JSON structure:

{
  "winner": "tie",
  "reason": "Comparative justification",
  "x_strength": "Strongest quality of NPC X",
  "x_weakness": "Main weakness of NPC X",
  "y_strength": "Strongest quality of NPC Y",
  "y_weakness": "Main weakness of NPC Y",
  "evidence_x": [
    {"turn": 1, "quote": "Exact NPC excerpt from X"}
  ],
  "evidence_y": [
    {"turn": 1, "quote": "Exact NPC excerpt from Y"}
  ]
}

Use only valid turn numbers and directly verifiable quotes.
Do not add fields.
"""


FIELDS = {
    "winner", "reason",
    "x_strength", "x_weakness",
    "y_strength", "y_weakness",
    "evidence_x", "evidence_y",
}


def validate_dataset(dataset):
    if (
        not isinstance(dataset, dict)
        or dataset.get("schema_version") != 1
        or not isinstance(dataset.get("cases"), list)
        or len(dataset["cases"]) < 2
    ):
        raise ValueError("Invalid pairwise dataset")

    ids = set()
    for case in dataset["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Invalid candidate")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in ids
        ):
            raise ValueError("Invalid or duplicate candidate ID")
        ids.add(identifier)

        transcript = case.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise ValueError("Invalid transcript")

        turns = {}
        for item in transcript:
            if (
                not isinstance(item, dict)
                or type(item.get("turn")) is not int
                or item["turn"] < 1
                or item.get("role") not in {"player", "npc"}
                or not isinstance(item.get("text"), str)
                or not item["text"].strip()
            ):
                raise ValueError("Invalid transcript observation")

            entry = turns.setdefault(item["turn"], set())
            if item["role"] in entry:
                raise ValueError("Duplicate observation role")
            entry.add(item["role"])

        if any(roles != {"player", "npc"} for roles in turns.values()):
            raise ValueError("Unpaired conversation turn")

    return dataset


def matchups(dataset, seed=1234):
    ids = sorted(case["id"] for case in validate_dataset(dataset)["cases"])
    pairs = list(itertools.combinations(ids, 2))
    jobs = []

    for a, b in pairs:
        jobs.extend([
            (a, b),
            (b, a),
        ])

    random.Random(seed).shuffle(jobs)
    return jobs


def validate_verdict(value, x, y):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Invalid pairwise fields")

    if value["winner"] not in {"X", "Y", "tie"}:
        raise ValueError("Invalid pairwise winner")

    for key in (
        "reason", "x_strength", "x_weakness",
        "y_strength", "y_weakness",
    ):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Missing justification: {key}")

    for key, case in (("evidence_x", x), ("evidence_y", y)):
        evidence = value[key]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Missing evidence: {key}")

        npc_turns = {
            item["turn"]: item["text"]
            for item in case["transcript"]
            if item["role"] == "npc"
        }

        for citation in evidence:
            if (
                not isinstance(citation, dict)
                or set(citation) != {"turn", "quote"}
                or type(citation["turn"]) is not int
                or not isinstance(citation["quote"], str)
                or not citation["quote"].strip()
                or citation["quote"] not in npc_turns.get(
                    citation["turn"], ""
                )
            ):
                raise ValueError(f"Unsupported NPC evidence: {key}")

    return value


def judge(x, y, context, backend, sampling):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": json.dumps({
            "character_and_scenario": context,
            "conversation_X": x["transcript"],
            "conversation_Y": y["transcript"],
        }, ensure_ascii=False)},
    ]

    url, payload = backend.prepare(messages, sampling, {})
    started = time.perf_counter()

    result = {
        "valid": False,
        "content": None,
        "reasoning": None,
        "finish_reason": None,
        "usage": None,
    }

    try:
        content, response = backend.generate(url, payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        result.update({
            "content": content,
            "reasoning": (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            ),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
        })

        if result["finish_reason"] == "length":
            raise ValueError("Pairwise generation truncated")

        proposal = parse_evaluator_json(content)
        validate_verdict(proposal, x, y)
        result["verdict"] = proposal
        result["valid"] = True

    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"

    result["elapsed_seconds"] = round(
        time.perf_counter() - started, 3
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--pair", nargs=2, metavar=("X", "Y")
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32768
    )
    args = parser.parse_args()

    if not 1024 <= args.max_tokens <= 65536:
        parser.error("max-tokens must be 1024..65536")

    dataset_bytes = Path(args.dataset).read_bytes()
    dataset = validate_dataset(json.loads(dataset_bytes))
    context = json.loads(
        Path(args.scenario).read_text(encoding="utf-8")
    )["system"]

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
        "max_tokens": args.max_tokens,
    }

    cases = {case["id"]: case for case in dataset["cases"]}
    jobs = matchups(dataset, args.seed)

    if args.pair is not None:
        xid, yid = args.pair
        if xid == yid or xid not in cases or yid not in cases:
            parser.error("Invalid candidate pair")
        jobs = [(xid, yid)]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation. Never overwrite archived results.
    with output.open("x", encoding="utf-8") as file:
        for index, (xid, yid) in enumerate(jobs, 1):
            print(
                f"Pair {index:02}/{len(jobs):02} | "
                f"{xid} vs {yid}",
                flush=True,
            )

            record = judge(
                cases[xid], cases[yid],
                context, backend, sampling,
            )

            record.update({
                "x": xid,
                "y": yid,
                "seed": args.seed,
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

            winner = (
                record["verdict"]["winner"]
                if record["valid"] else "INVALID"
            )
            print(
                f"  winner={winner} "
                f"valid={record['valid']} "
                f"tokens={(record.get('usage') or {}).get('completion_tokens')}",
                flush=True,
            )

    print("Saved:", output)


if __name__ == "__main__":
    main()
