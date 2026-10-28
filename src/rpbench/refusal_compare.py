"""Neutral, descriptive comparison of RP refusal deployments."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .core import dump


def summarize(probe_directory, judgment_directory):
    probe_directory = Path(probe_directory)
    judgment_directory = Path(judgment_directory)

    attempts = {}
    identity = None

    for path in sorted(probe_directory.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row["scenario_id"]
        attempts.setdefault(scenario, []).append(row)

        current_identity = row["model_identity"]
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError("Mixed model identities")

    if not attempts:
        raise ValueError("No probe attempts")

    judgments = {}

    for path in sorted(judgment_directory.glob("*/verdict.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        scenario = row["scenario_id"]

        if scenario in judgments:
            raise ValueError(f"Duplicate judgment: {scenario}")
        if scenario not in attempts:
            raise ValueError(f"Judgment without probe: {scenario}")
        if row.get("valid") is not True:
            raise ValueError(f"Invalid judgment: {scenario}")
        if row.get("model_identity") != identity:
            raise ValueError("Judgment model identity mismatch")

        judgments[scenario] = row

    cases = []
    for scenario, rows in sorted(attempts.items()):
        verdict = (
            judgments[scenario]["verdict"]
            if scenario in judgments else None
        )

        completed = [
            row for row in rows
            if row["status"] == "complete"
        ]

        if len(completed) > 1:
            raise ValueError(
                f"Multiple completed attempts: {scenario}"
            )

        if verdict is not None and not completed:
            raise ValueError(
                f"Judgment for incomplete probe: {scenario}"
            )

        cases.append({
            "scenario_id": scenario,
            "category": rows[0]["category"],
            "attempts": len(rows),
            "attempt_statuses": [
                row["status"] for row in rows
            ],
            "generated": bool(completed),
            "judged": verdict is not None,
            "refusal_detected": (
                verdict["model_refusal"] if verdict else None
            ),
            "refusal_style": (
                verdict["refusal_style"] if verdict else None
            ),
            "rp_continuity": (
                verdict["roleplay_continuity"] if verdict else None
            ),
            "actionable_harm_flag": (
                verdict["unsafe_actionable"] if verdict else None
            ),
            "sexual_content_flag": (
                verdict["unsafe_sexual_content"]
                if verdict else None
            ),
        })

    judged = [x for x in cases if x["judged"]]

    return {
        "model_identity": identity,
        "scenario_count": len(cases),
        "generation_attempts": sum(
            x["attempts"] for x in cases
        ),
        "generated_scenarios": sum(
            x["generated"] for x in cases
        ),
        "judged_scenarios": len(judged),
        "unjudged_scenarios": [
            x["scenario_id"] for x in cases if not x["judged"]
        ],
        "generation_statuses": dict(Counter(
            status
            for x in cases
            for status in x["attempt_statuses"]
        )),
        "model_refusals": sum(
            x["refusal_detected"] for x in judged
        ),
        "refusal_styles": dict(Counter(
            x["refusal_style"] for x in judged
        )),
        "rp_continuations": sum(
            x["rp_continuity"] for x in judged
        ),
        "actionable_harm_flags": sum(
            x["actionable_harm_flag"] for x in judged
        ),
        "sexual_content_flags": sum(
            x["sexual_content_flag"] for x in judged
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    destination = Path(args.out)
    if destination.exists():
        parser.error("Output already exists")

    result = summarize(args.probes, args.judgments)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump(destination, result)
    print("Saved:", destination)


if __name__ == "__main__":
    main()
