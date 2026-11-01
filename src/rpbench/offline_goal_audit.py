"""Independent, evidence-verified, full-transcript goal audit v2.

Does not edit source runs or consume live judge assessments.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .core import Backend, dump, load
from .evaluator_llm import parse_evaluator_json
from .observations import read_observations
from .goal_criteria import validate_goal_criteria
from .state import initial_state


SCHEMA_VERSION = 2
STATUSES = {"satisfied", "partial", "not_satisfied"}

INSTRUCTIONS = """You are an independent auditor of a fictional
roleplay transcript. Evaluate ONE goal using the full conversation.

Judge actual observable events, not the speed of quest completion.
Do not reward rapid progress or penalize slow narrative development.

The transcript may contain different documents and timestamps.
Never conflate an archive visitor register, a key ledger, an
access list, or a reception note. An exact match of date or time
does not imply that two records are the same document.

Use both player actions and NPC replies as evidence.
An action described by the player does not prove an NPC complied.
A character's statement is not automatically objective truth.
Infer no event merely from the scenario's hidden world facts.

Evaluate EVERY supplied criterion independently. A criterion
can become satisfied in any turn, including a later turn.
Do not treat previous live-judge decisions as authoritative.

Return ONLY a JSON object:
{
  "schema_version": 2,
  "goal_id": "supplied goal ID",
  "criteria": [
    {
      "criterion_id": "supplied criterion ID",
      "status": "satisfied",
      "first_satisfied_turn": 7,
      "evidence": [
        {
          "turn": 7,
          "role": "npc",
          "quote": "EXACT contiguous substring from that turn"
        }
      ],
      "reason": "Explain every essential requirement and distinguish sources"
    }
  ]
}

For partial, use first_satisfied_turn=null and cite concrete
evidence for a necessary but incomplete component.
For not_satisfied, use first_satisfied_turn=null and
evidence=[] unless a quote is needed to explain a clear failure.
For satisfied, first_satisfied_turn must be the earliest turn at
which all required evidence was observable, not just the final turn.

Evidence quotes must be exact nonempty contiguous substrings from
the identified player or NPC turn. No ellipses, paraphrases,
fabricated turns, additional keys or Markdown.
"""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_verdict(verdict, goal_id, definition, observations):
    if not isinstance(verdict, dict) or set(verdict) != {
        "schema_version", "goal_id", "criteria"
    }:
        raise ValueError("Invalid audit object")

    if (
        type(verdict["schema_version"]) is not int
        or verdict["schema_version"] != SCHEMA_VERSION
        or verdict["goal_id"] != goal_id
    ):
        raise ValueError("Invalid audit version or goal")

    expected = {
        item["id"]: item
        for item in definition["goals"][goal_id]["criteria"]
    }
    items = verdict["criteria"]
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Incomplete audit criteria")

    lookup = {
        (item["turn"], item["role"]): item["text"]
        for item in observations
    }
    last_turn = max(item["turn"] for item in observations)
    seen = set()
    cleaned = []

    for item in items:
        required = {
            "criterion_id", "status", "first_satisfied_turn",
            "evidence", "reason",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("Invalid criterion audit fields")

        cid = item["criterion_id"]
        status = item["status"]
        first = item["first_satisfied_turn"]
        evidence = item["evidence"]
        reason = item["reason"]

        if cid not in expected or cid in seen:
            raise ValueError("Unknown or duplicated criterion")
        seen.add(cid)

        if status not in STATUSES:
            raise ValueError("Invalid criterion audit status")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Missing criterion reasoning")
        if (
            not isinstance(evidence, list)
            or len(evidence) > 12
            or (status in {"satisfied", "partial"} and not evidence)
        ):
            raise ValueError("Missing or excessive audit evidence")

        if status == "satisfied":
            if (
                type(first) is not int
                or not 1 <= first <= last_turn
            ):
                raise ValueError("Invalid first satisfied turn")
        elif first is not None:
            raise ValueError("Only satisfied criteria have completion turns")

        clean_evidence = []
        signatures = set()

        for ref in evidence:
            if not isinstance(ref, dict) or set(ref) != {
                "turn", "role", "quote"
            }:
                raise ValueError("Invalid evidence reference")

            turn = ref["turn"]
            role = ref["role"]
            quote = ref["quote"]

            if (
                type(turn) is not int
                or role not in {"player", "npc"}
                or (turn, role) not in lookup
                or not isinstance(quote, str)
                or not quote.strip()
                or quote not in lookup[(turn, role)]
            ):
                raise ValueError("Audit evidence source mismatch")

            signature = (turn, role, quote)
            if signature in signatures:
                raise ValueError("Duplicate audit evidence")
            signatures.add(signature)
            clean_evidence.append(dict(ref))

        if status == "satisfied" and (
            max(ref["turn"] for ref in clean_evidence) > first
        ):
            raise ValueError(
                "Completion time precedes cited essential evidence"
            )

        cleaned.append({
            "criterion_id": cid,
            "status": status,
            "first_satisfied_turn": first,
            "evidence": clean_evidence,
            "reason": reason.strip(),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "goal_id": goal_id,
        "criteria": sorted(
            cleaned, key=lambda item: list(expected).index(
                item["criterion_id"]
            )
        ),
    }


def prepare(run, goal_id, model, base_url, max_tokens):
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError("Missing source run")

    manifest = load(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Source run is not complete")

    ground_truth = load(run / "ground_truth.json")
    definition = manifest["runtime"]["goal_criteria"]
    validate_goal_criteria(
        definition, initial_state(ground_truth)
    )

    if goal_id not in definition["goals"]:
        raise ValueError(f"Unknown goal: {goal_id}")

    observations = read_observations(run)

    source_filenames = (
        "manifest.json", "ground_truth.json",
        "observations.jsonl", "events.jsonl",
    )
    source_hashes = {
        name: sha256_bytes((run / name).read_bytes())
        for name in source_filenames
        if (run / name).is_file()
    }

    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_run": str(run),
        "source_files_sha256": source_hashes,
        "goal_id": goal_id,
        "definition_sha256": sha256_bytes(encode(definition)),
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "instructions_sha256": sha256_bytes(
            INSTRUCTIONS.encode("utf-8")
        ),
    }
    request_context = {
        "goal_id": goal_id,
        "criteria": [
            {
                "criterion_id": item["id"],
                "description": item["description"],
            }
            for item in definition["goals"][goal_id]["criteria"]
        ],
        "transcript": [
            {
                "turn": item["turn"],
                "role": item["role"],
                "text": item["text"],
            }
            for item in observations
        ],
    }
    return identity, definition, observations, request_context


def audit(
    run, output, goal_id, model,
    base_url="https://nano-gpt.com/api/subscription",
    max_tokens=16384, dry_run=False,
):
    source = Path(run).resolve()
    output = Path(output).resolve()

    if output == source or source in output.parents:
        raise ValueError("Audit output must be outside the source run")

    if not model or not base_url or max_tokens < 1:
        raise ValueError("Invalid audit backend configuration")

    identity, definition, observations, context = prepare(
        source, goal_id, model, base_url, max_tokens
    )

    if dry_run:
        print("Goal:", goal_id)
        print("Model:", model)
        print("Transcript turns:", max(
            item["turn"] for item in observations
        ))
        print("Criteria:", len(context["criteria"]))
        print("Endpoint:", base_url.rstrip("/") + "/v1/chat/completions")
        return None

    identity_file = output / "identity.json"
    result_file = output / "verdict.json"

    if identity_file.exists():
        if load(identity_file) != identity:
            raise ValueError("Audit identity mismatch")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty uninitialized audit output")

    if result_file.exists():
        result = load(result_file)
        validate_verdict(
            result["verdict"], goal_id, definition, observations
        )
        if result.get("identity_sha256") != sha256_bytes(
            encode(identity)
        ):
            raise ValueError("Existing audit verdict identity mismatch")
        print("Existing validated audit:", result_file)
        return result

    if not os.getenv("NANOGPT_API_KEY"):
        raise ValueError("Missing NANOGPT_API_KEY")

    backend = Backend(
        name="nanogpt",
        base_url=base_url,
        model=model,
        mode="chat",
        api_key=os.getenv("NANOGPT_API_KEY"),
        timeout=300,
    )

    url, payload = backend.prepare(
        [
            {"role": "system", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(
                    context, ensure_ascii=False
                ),
            },
        ],
        {
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        {},
    )

    # Write identity only after response validation succeeds.
    # A failed API call leaves no accepted verdict.
    content, response = backend.generate(url, payload)
    choice = (response.get("choices") or [{}])[0]

    if choice.get("finish_reason") == "length":
        raise ValueError("Offline judge exhausted its output budget")

    proposal = parse_evaluator_json(content)
    verdict = validate_verdict(
        proposal, goal_id, definition, observations
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": sha256_bytes(encode(identity)),
        "verdict": verdict,
        "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }

    output.mkdir(parents=True, exist_ok=True)
    if not identity_file.exists():
        dump(identity_file, identity)
    dump(result_file, result)

    print("Audit saved:", result_file)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default="https://nano-gpt.com/api/subscription",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit(
        args.run, args.out, args.goal, args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
