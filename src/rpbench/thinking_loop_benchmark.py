"""Compare reasoning repetition across evaluator sampling profiles."""

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .core import load
from .criteria_runtime import _generate
from .evaluator_llm import evaluate_llm
from .evaluator_runtime import validate_evaluator_config
from .knowledge_gate import observable_text, split_paragraphs
from .knowledge_gate_journal import read_gate_events
from .knowledge_normalization import normalize_knowledge_proposal
from .observations import read_jsonl
from .player_evaluator import validate_evaluation
from .semantic_benchmark import case_inputs
from .semantic_criteria import validate_semantic_proposal
from .state import initial_state


def repetition_metrics(reasoning, window=16):
    """Measure repeated word windows, not merely output length."""
    words = re.findall(r"\b[\w'-]+\b", reasoning.casefold())

    if len(words) < window * 2:
        return {
            "words": len(words),
            "repeated_window_fraction": 0.0,
            "max_window_occurrences": 0,
        }

    windows = [
        tuple(words[i:i + window])
        for i in range(len(words) - window + 1)
    ]
    counts = Counter(windows)

    # Ignore the first occurrence of every repeated window.
    repeated = sum(count - 1 for count in counts.values())

    return {
        "words": len(words),
        "repeated_window_fraction": round(
            repeated / len(windows), 4
        ),
        "max_window_occurrences": max(counts.values()),
    }


def make_knowledge_case(run):
    observations = read_jsonl(run / "observations.jsonl")
    gates = read_gate_events(
        run / "events.jsonl", observations
    )

    player = next(
        item["text"] for item in observations
        if item["turn"] == 7 and item["role"] == "player"
    )
    npc = next(
        item["text"] for item in observations
        if item["turn"] == 7 and item["role"] == "npc"
    )
    visible = observable_text(
        split_paragraphs(npc), gates[7]["labels"]
    )
    state = initial_state(load(run / "ground_truth.json"))

    return player, npc, visible, state


def run_one(settings, task, case, knowledge_data):
    capture = {}
    start = time.perf_counter()

    record = {
        "task": task,
        "model": settings["backend"]["model"],
        "sampling": dict(settings["sampling"]),
    }

    try:
        if task == "knowledge":
            player, npc, visible, state = knowledge_data
            request_settings = {
                **settings,
                "_source_turn": 7,
                "_player_action": player,
            }

            proposal = evaluate_llm(
                visible,
                state,
                {"facts": []},
                request_settings,
                knowledge_only=True,
                on_response=capture.update,
            )
            normalized = normalize_knowledge_proposal(
                proposal, visible, 7
            )
            validated = validate_evaluation(
                normalized, npc, state, turn=7
            )
            record["knowledge_facts"] = len(
                validated["knowledge_updates"]
            )
            record["semantically_scored"] = False
        else:
            definition, state, progress = case_inputs(case)
            proposal, metadata = _generate(
                definition,
                progress,
                case["observations"],
                settings,
                goal_id=case["goal_id"],
                criterion_ids=[case["criterion_id"]],
                on_response=capture.update,
            )
            turn = max(
                item["turn"] for item in case["observations"]
            )
            validated = validate_semantic_proposal(
                proposal, definition, state,
                case["observations"], turn,
            )
            update = validated["criterion_updates"]
            if len(update) != 1:
                raise ValueError("Unexpected criterion update count")
            record["status"] = update[0]["status"]
            record["correct"] = (
                update[0]["status"] == case["expected_status"]
            )
            record["semantically_scored"] = True

        if capture.get("finish_reason") == "length":
            raise ValueError("Generation truncated")

        record["valid"] = True

    except Exception as error:
        record["valid"] = False
        record["error"] = f"{type(error).__name__}: {error}"

    record["elapsed_seconds"] = round(
        time.perf_counter() - start, 3
    )
    record["finish_reason"] = capture.get("finish_reason")
    record["usage"] = capture.get("usage")
    record["content"] = capture.get("content", "")
    record["reasoning"] = capture.get("reasoning", "")

    reasoning = record["reasoning"] or ""
    record.update(repetition_metrics(reasoning))

    record["reasoning_only_truncation"] = (
        record["finish_reason"] == "length"
        and not (record["content"] or "").strip()
        and bool(reasoning)
    )
    # Heuristic signal; inspect the saved trace before
    # declaring a genuine thinking loop.
    record["repetition_signal"] = (
        record["repeated_window_fraction"] >= 0.35
        and record["max_window_occurrences"] >= 4
        and record["words"] >= 300
    )

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not 1 <= args.repetitions <= 10:
        parser.error("repetitions must be 1..10")
    if not 256 <= args.max_tokens <= 32768:
        parser.error("max-tokens must be 256..32768")
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be 1..3600")

    base = validate_evaluator_config(
        load(args.evaluator_config)
    )
    if base is None or base["mode"] != "llm":
        parser.error("LLM evaluator required")

    suite = load(args.suite)
    case = next(
        c for c in suite["cases"]
        if c["id"] == "voss-retracted"
    )
    knowledge_data = make_knowledge_case(Path(args.run))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Refuse accidental overwrite of an earlier experiment.
    with output.open("x", encoding="utf-8") as file:
        for repetition in range(1, args.repetitions + 1):
            for temperature in (0.0, 1.0):
                settings = {
                    **base,
                    "backend": {
                        **base["backend"],
                        "timeout": args.timeout,
                    },
                    "sampling": {
                        **base["sampling"],
                        "temperature": temperature,
                        "max_tokens": args.max_tokens,
                    },
                }

                for task in ("knowledge", "criteria"):
                    print(
                        f"{task} | temp={temperature} "
                        f"| repetition={repetition}",
                        flush=True,
                    )
                    record = run_one(
                        settings, task, case, knowledge_data
                    )
                    record.update({
                        "repetition": repetition,
                        "timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    })
                    file.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )
                    file.flush()

                    print(
                        f"  valid={record['valid']} "
                        f"tokens={(record.get('usage') or {}).get('completion_tokens')} "
                        f"repeated={record['repeated_window_fraction']:.3f} "
                        f"loop_signal={record['repetition_signal']} "
                        f"truncated={record['reasoning_only_truncation']}",
                        flush=True,
                    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
