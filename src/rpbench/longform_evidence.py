"""Validate quoted long-form RP consistency evidence against archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .observations import read_observations


CATEGORIES = {
    "world_state",
    "timeline",
    "player_agency",
    "knowledge_boundary",
    "character_development",
    "narrative_repetition",
}

VERDICTS = {"confirmed", "candidate", "no_issue"}


def validate_finding(finding, observations):
    if not isinstance(finding, dict):
        raise ValueError("Finding must be an object")

    category = finding.get("category")
    verdict = finding.get("verdict")

    if category not in CATEGORIES:
        raise ValueError("Invalid finding category")

    if verdict not in VERDICTS:
        raise ValueError("Invalid finding verdict")

    reason = finding.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Finding requires a reason")

    evidence = finding.get("evidence")

    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Finding requires evidence")

    lookup = {
        (entry["turn"], entry["role"]): entry["text"]
        for entry in observations
    }

    positions = []

    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Invalid evidence item")

        turn = item.get("turn")
        role = item.get("role")
        quote = item.get("quote")

        if type(turn) is not int or turn < 1:
            raise ValueError("Invalid evidence turn")

        if role not in {"player", "npc"}:
            raise ValueError("Invalid evidence role")

        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Empty evidence quote")

        text = lookup.get((turn, role))
        if text is None or quote not in text:
            raise ValueError(
                f"Evidence not found in {role} turn {turn}"
            )

        positions.append((turn, 0 if role == "player" else 1))

    if positions != sorted(positions):
        raise ValueError("Evidence must be chronological")

    if (
        category == "character_development"
        and verdict == "confirmed"
        and len(set(turn for turn, _ in positions)) < 3
    ):
        raise ValueError(
            "Confirmed character drift needs evidence "
            "from at least three distinct turns"
        )


def validate_report(run_dir, report):
    observations = read_observations(run_dir)

    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
    ):
        raise ValueError("Invalid evidence report schema")

    findings = report.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Findings must be a list")

    for finding in findings:
        validate_finding(finding, observations)

    return {
        "findings": len(findings),
        "confirmed": sum(
            f["verdict"] == "confirmed" for f in findings
        ),
        "candidates": sum(
            f["verdict"] == "candidate" for f in findings
        ),
        "no_issue": sum(
            f["verdict"] == "no_issue" for f in findings
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = json.loads(
        Path(args.report).read_text(encoding="utf-8")
    )

    print(json.dumps(
        validate_report(args.run, report),
        indent=2,
    ))


if __name__ == "__main__":
    main()
