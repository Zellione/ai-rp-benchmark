"""Retrospective false-positive audit at historical goal completion turns.

Read-only with respect to source runs. Audit files live separately.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .core import Backend, dump, load
from .criteria_progress import advance_criteria_goals
from .evaluator_llm import parse_evaluator_json
from .goal_criteria import validate_goal_criteria
from .observations import read_jsonl, read_observations
from .semantic_criteria import EVENT_TYPE
from .state import initial_state


INSTRUCTIONS = """
You independently audit an RP simulation's ONLINE goal completion.

Only the supplied transcript prefix existed when the online judge
marked this goal completed. Never infer evidence from later turns.

Evaluate whether EVERY listed criterion was already satisfied
by the END of the supplied completion turn.

Interpret narrative evidence semantically. Paraphrases are allowed.
A player claim does not automatically prove that an NPC complied.
An intended, requested, or promised future action is not completed.
Do not infer observations from hidden scenario facts.

Use four criterion statuses:
- satisfied: the actual required event is supported
- partial: a necessary component is supported, but not all
- not_satisfied: essential evidence is missing or contradicted
- uncertain: the transcript cannot support a confident judgment

Do not use earlier online-judge decisions as authoritative.

Return ONLY a JSON object:
{
  "schema_version": 1,
  "goal_id": "supplied goal ID",
  "criteria": [
    {
      "criterion_id": "supplied criterion ID",
      "status": "satisfied",
      "evidence": [
        {
          "turn": 1,
          "role": "npc",
          "summary": "Brief semantic description of evidence"
        }
      ],
      "reason": "Explain the judgment"
    }
  ]
}

Include each supplied criterion exactly once.
Evidence references must point to existing turns and roles.
No exact quotations are required.
"""


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def find_completions(definition, original, observations, events):
    """Find first online-completed turn of each goal."""
    semantic = [
        item for item in events
        if item.get("type") == EVENT_TYPE
    ]

    previous = original
    completions = []

    turns = sorted({
        item["turn"]
        for item in observations
        if item["role"] == "npc"
    })

    for turn in turns:
        prefix = [
            item for item in observations
            if item["turn"] <= turn
        ]
        historical = [
            item for item in semantic
            if item["turn"] <= turn
        ]

        current, _, _ = advance_criteria_goals(
            original, definition, prefix, historical
        )

        for goal_id in definition["goals"]:
            before = previous["goals"][goal_id]["status"]
            after = current["goals"][goal_id]["status"]

            if before != "completed" and after == "completed":
                completions.append({
                    "turn": turn,
                    "goal_id": goal_id,
                })

        previous = current

    return sorted(
        completions,
        key=lambda item: (item["turn"], item["goal_id"]),
    )


def validate_verdict(verdict, goal_id, definition, prefix):
    """Validate format and source provenance, not exact wording."""
    expected = {
        item["id"]
        for item in definition["goals"][goal_id]["criteria"]
    }
    sources = {
        (item["turn"], item["role"])
        for item in prefix
    }

    if (
        not isinstance(verdict, dict)
        or set(verdict) != {
            "schema_version", "goal_id", "criteria"
        }
        or type(verdict["schema_version"]) is not int
        or verdict["schema_version"] != 1
        or verdict["goal_id"] != goal_id
    ):
        raise ValueError("Invalid audit schema")

    items = verdict["criteria"]

    if not isinstance(items, list):
        raise ValueError("Invalid criteria list")

    if {
        item.get("criterion_id")
        for item in items
        if isinstance(item, dict)
    } != expected or len(items) != len(expected):
        raise ValueError("Audit criterion mismatch")

    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "criterion_id", "status", "evidence", "reason"
        }:
            raise ValueError("Invalid criterion fields")

        if item["status"] not in {
            "satisfied", "partial", "not_satisfied", "uncertain"
        }:
            raise ValueError("Invalid criterion status")

        if (
            not isinstance(item["reason"], str)
            or not item["reason"].strip()
        ):
            raise ValueError("Missing reasoning")

        evidence = item["evidence"]

        if not isinstance(evidence, list) or len(evidence) > 12:
            raise ValueError("Invalid evidence list")

        for ref in evidence:
            if not isinstance(ref, dict) or set(ref) != {
                "turn", "role", "summary"
            }:
                raise ValueError("Invalid evidence reference")

            if (
                type(ref["turn"]) is not int
                or (ref["turn"], ref["role"]) not in sources
                or not isinstance(ref["summary"], str)
                or not ref["summary"].strip()
            ):
                raise ValueError("Invalid evidence provenance")

    return verdict


def classify(verdict):
    statuses = [
        item["status"]
        for item in verdict["criteria"]
    ]

    if any(status == "uncertain" for status in statuses):
        return "uncertain"

    if all(status == "satisfied" for status in statuses):
        return "valid"

    return "false_positive"


def audit_one(
    run, output, manifest, definition, observations,
    candidate, model, base_url, max_tokens,
):
    goal_id = candidate["goal_id"]
    turn = candidate["turn"]

    prefix = [
        {
            "turn": item["turn"],
            "role": item["role"],
            "text": item["text"],
        }
        for item in observations
        if item["turn"] <= turn
    ]

    context = {
        "goal_id": goal_id,
        "completion_turn": turn,
        "criteria": [
            {
                "criterion_id": item["id"],
                "description": item["description"],
            }
            for item in definition["goals"][goal_id]["criteria"]
        ],
        "transcript": prefix,
    }

    identity = {
        "source_run": str(run.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            (run / "manifest.json").read_bytes()
        ).hexdigest(),
        "context_sha256": digest(context),
        "instructions_sha256": hashlib.sha256(
            INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
    }

    identity_file = output / "identity.json"
    raw_file = output / "raw_response.json"
    verdict_file = output / "verdict.json"

    if identity_file.exists():
        if load(identity_file) != identity:
            raise ValueError("Audit identity mismatch")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Unexpected existing audit files")

    if verdict_file.exists():
        result = load(verdict_file)
        validate_verdict(
            result["verdict"], goal_id, definition, prefix
        )
        if result.get("identity_sha256") != digest(identity):
            raise ValueError("Stored verdict identity mismatch")
        return result

    if raw_file.exists():
        raw = load(raw_file)
        if raw.get("identity_sha256") != digest(identity):
            raise ValueError("Raw response identity mismatch")
    else:
        api_key = os.getenv("NANOGPT_API_KEY")
        if not api_key:
            raise ValueError("Missing NANOGPT_API_KEY")

        backend = Backend(
            name="nanogpt",
            base_url=base_url,
            model=model,
            mode="chat",
            api_key=api_key,
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
            {"temperature": 0, "max_tokens": max_tokens},
            {},
        )

        content, response = backend.generate(url, payload)

        if not isinstance(content, str):
            raise ValueError("Invalid auditor response")

        # Preserve the entire provider response, including any
        # reasoning_content actually returned by the provider.
        raw = {
            "identity_sha256": digest(identity),
            "content": content,
            "response": response,
        }

        output.mkdir(parents=True, exist_ok=True)
        dump(identity_file, identity)
        dump(raw_file, raw)

    response = raw["response"]
    choice = (response.get("choices") or [{}])[0]

    if choice.get("finish_reason") == "length":
        raise ValueError("Auditor exhausted output tokens")

    proposal = parse_evaluator_json(raw["content"])
    verdict = validate_verdict(
        proposal, goal_id, definition, prefix
    )

    result = {
        "identity_sha256": digest(identity),
        "goal_id": goal_id,
        "completion_turn": turn,
        "restart_player_turn": turn + 1,
        "classification": classify(verdict),
        "verdict": verdict,
        "usage": response.get("usage"),
    }

    dump(verdict_file, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v4.1-flash:thinking",
    )
    parser.add_argument(
        "--base-url",
        default="https://nano-gpt.com/api",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-checks", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_checks < 1 or args.max_tokens < 1:
        parser.error("Invalid audit limits")

    run = Path(args.run).resolve()
    output = Path(args.out).resolve()

    if output == run or run in output.parents:
        raise ValueError("Audit output must be outside source run")

    manifest = load(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Source simulation must be complete")

    definition = manifest["runtime"]["goal_criteria"]
    original = initial_state(load(run / "ground_truth.json"))
    definition = validate_goal_criteria(definition, original)

    observations = read_observations(run)
    events = read_jsonl(run / "events.jsonl")

    candidates = find_completions(
        definition, original, observations, events
    )

    print("=== ONLINE GOAL COMPLETIONS ===")
    for item in candidates:
        print(
            f"Turn {item['turn']:2d}: {item['goal_id']}"
        )

    if not args.execute:
        print("\nDry-run only. Add --execute to audit.")
        return

    checked = 0
    earliest_false_positive = None

    for candidate in candidates:
        if checked >= args.max_checks:
            break

        subdir = output / (
            f"turn-{candidate['turn']:03d}-"
            f"{candidate['goal_id']}"
        )

        print("\nAuditing:", subdir.name, flush=True)

        result = audit_one(
            run, subdir, manifest, definition,
            observations, candidate,
            args.model, args.base_url, args.max_tokens,
        )

        checked += 1

        print("Classification:", result["classification"])

        if result["classification"] == "false_positive":
            earliest_false_positive = result
            print(
                "RESTART CANDIDATE:",
                result["restart_player_turn"],
            )
            break

        if result["classification"] == "uncertain":
            print("Manual review required; stopping audit")
            break

    if earliest_false_positive:
        print("\n=== EARLIEST CONFIRMED FALSE POSITIVE ===")
        print(json.dumps({
            "goal_id": earliest_false_positive["goal_id"],
            "completion_turn": (
                earliest_false_positive["completion_turn"]
            ),
            "restart_player_turn": (
                earliest_false_positive["restart_player_turn"]
            ),
        }, indent=2))
    else:
        print(
            "\nNo confirmed false positive in the "
            "goals checked by this invocation."
        )


if __name__ == "__main__":
    main()
