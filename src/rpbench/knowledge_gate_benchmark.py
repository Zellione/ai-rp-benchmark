"""Isolated player-perspective gate benchmark."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import load
from .evaluator_runtime import validate_evaluator_config
from .knowledge_gate import evaluate_gate
from .thinking_loop_benchmark import repetition_metrics


CASES = [
    {
        "id": "observable-conversation",
        "player_action": (
            "Elena stands beside Mara in the archive "
            "and listens to her explanation."
        ),
        "npc_text": (
            "Mara tells Elena that the public reading room "
            "is open until six o'clock."
        ),
        "expected": ["observable"],
    },
    {
        "id": "private-offscreen",
        "player_action": (
            "Elena remains in the public reading room, "
            "far from Voss's closed office."
        ),
        "npc_text": (
            "Inside his closed office, Voss privately tells "
            "Mara that the restricted vault key is hidden "
            "behind the blue lantern. Elena cannot hear them."
        ),
        "expected": ["offscreen"],
    },
    {
        "id": "mixed-scene",
        "player_action": (
            "Elena asks Mara about archive opening hours "
            "and prepares to leave."
        ),
        "npc_text": (
            "Mara tells Elena that the public reading room "
            "closes at six o'clock.\n\n"
            "Elena thanks Mara and leaves the archive.\n\n"
            "After Elena leaves, Voss privately tells Mara "
            "that the restricted vault key is hidden "
            "behind the blue lantern."
        ),
        "expected": [
            "observable",
            "observable",
            "offscreen",
        ],
    },
]


def run_case(case, settings):
    capture = {}
    start = time.perf_counter()

    result = {
        "case": case["id"],
        "expected": case["expected"],
    }

    try:
        labels, metadata = evaluate_gate(
            case["player_action"],
            case["npc_text"],
            settings,
            on_response=capture.update,
        )

        result.update({
            "valid": True,
            "labels": labels,
            "correct": labels == case["expected"],
        })

    except Exception as error:
        result.update({
            "valid": False,
            "correct": False,
            "error": f"{type(error).__name__}: {error}",
        })

    result.update({
        "elapsed_seconds": round(
            time.perf_counter() - start, 3
        ),
        "usage": capture.get("usage"),
        "finish_reason": capture.get("finish_reason"),
        "content": capture.get("content"),
        "reasoning": capture.get("reasoning"),
    })

    reasoning = result["reasoning"] or ""
    result.update(repetition_metrics(reasoning))

    result["reasoning_only_truncation"] = (
        result["finish_reason"] == "length"
        and not (result["content"] or "").strip()
        and bool(reasoning)
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark player-perspective scene gating"
    )
    parser.add_argument("--evaluator-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    if not 1 <= args.repetitions <= 20:
        parser.error("repetitions must be 1..20")

    config_bytes = Path(args.evaluator_config).read_bytes()
    settings = validate_evaluator_config(
        json.loads(config_bytes)
    )

    if settings is None or settings["mode"] != "llm":
        parser.error("LLM evaluator required")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        for repetition in range(1, args.repetitions + 1):
            for case in CASES:
                print(
                    f"{case['id']} | repetition {repetition}",
                    flush=True,
                )

                result = run_case(case, settings)
                result.update({
                    "repetition": repetition,
                    "model": settings["backend"]["model"],
                    "sampling": settings["sampling"],
                    "evaluator_config_sha256": hashlib.sha256(
                        config_bytes
                    ).hexdigest(),
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

                file.write(
                    json.dumps(result, ensure_ascii=False)
                    + "\n"
                )
                file.flush()

                print(
                    f"  labels={result.get('labels')} "
                    f"correct={result['correct']} "
                    f"valid={result['valid']} "
                    f"tokens={(result.get('usage') or {}).get('completion_tokens')} "
                    f"time={result['elapsed_seconds']}s",
                    flush=True,
                )

    print("Saved:", output)


if __name__ == "__main__":
    main()
