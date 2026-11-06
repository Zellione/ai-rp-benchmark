"""Audit, confirm and repair false-positive goal completions.

The source simulation is immutable. Every repair uses a new branch.
"""

import argparse
import json
from pathlib import Path

from .core import load, dump
from .false_positive_audit import (
    audit_one,
    find_completions,
)
from .false_positive_branch_run import prepare_branch
from .false_positive_branch import rejected_criteria
from .goal_criteria import validate_goal_criteria
from .observations import read_jsonl, read_observations
from .resume import resume_simulation
from .state import initial_state


def inspect_run(root):
    root = Path(root).resolve()
    manifest = load(root / "manifest.json")

    if (
        manifest.get("type") != "goal_driven_selfplay"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Expected a completed self-play run")

    original = initial_state(
        load(root / "ground_truth.json")
    )
    definition = validate_goal_criteria(
        manifest["runtime"]["goal_criteria"],
        original,
    )

    observations = read_observations(root)
    events = read_jsonl(root / "events.jsonl")

    candidates = find_completions(
        definition,
        original,
        observations,
        events,
    )

    return manifest, definition, observations, candidates


def audit_cycle(
    source,
    audit_root,
    model,
    confirmation_model,
    base_url,
    max_tokens,
):
    manifest, definition, observations, candidates = (
        inspect_run(source)
    )

    records = []

    for candidate in candidates:
        turn = candidate["turn"]
        goal_id = candidate["goal_id"]
        name = f"turn-{turn:03d}-{goal_id}"

        result = audit_one(
            source,
            audit_root / name,
            manifest,
            definition,
            observations,
            candidate,
            model,
            base_url,
            max_tokens,
        )

        status = result["classification"]

        record = {
            "turn": turn,
            "goal_id": goal_id,
            "classification": status,
            "audit_file": str(
                audit_root / name / "verdict.json"
            ),
        }
        records.append(record)

        print(
            f"Turn {turn:03d} | {goal_id} | {status}",
            flush=True,
        )

        if status == "uncertain":
            return "uncertain", records

        if status == "false_positive":
            if not confirmation_model:
                return "needs_confirmation", records

            confirmed = audit_one(
                source,
                audit_root.parent / (
                    audit_root.name + "-confirmation"
                ) / name,
                manifest,
                definition,
                observations,
                candidate,
                confirmation_model,
                base_url,
                max_tokens,
            )

            confirmation = confirmed["classification"]

            record["confirmation"] = confirmation

            if confirmation == "uncertain":
                return "uncertain", records

            if confirmation != "false_positive":
                return "disputed", records

            primary_rejected = rejected_criteria(
                result["verdict"]
            )
            confirmation_rejected = rejected_criteria(
                confirmed["verdict"]
            )

            record["rejected_criteria"] = sorted(
                primary_rejected
            )
            record["confirmation_rejected_criteria"] = sorted(
                confirmation_rejected
            )

            if primary_rejected != confirmation_rejected:
                return "disputed", records

            record["confirmed"] = True
            return "false_positive", records

    return "valid", records


def run_workflow(
    source,
    workspace,
    *,
    model="deepseek/deepseek-v4.1-flash:thinking",
    confirmation_model=None,
    base_url="https://nano-gpt.com/api",
    max_tokens=32768,
    max_repairs=2,
):
    source = Path(source).resolve()
    workspace = Path(workspace).resolve()

    if (
        workspace == source
        or source in workspace.parents
        or workspace in source.parents
    ):
        raise ValueError(
            "Workspace must be separate from source run"
        )

    if (
        type(max_repairs) is not int
        or not 0 <= max_repairs <= 2
        or type(max_tokens) is not int
        or max_tokens < 1
    ):
        raise ValueError("Invalid workflow limits")

    if (
        confirmation_model is not None
        and confirmation_model == model
    ):
        raise ValueError(
            "Confirmation requires a different judge model"
        )

    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(
            "Workflow workspace must be empty"
        )

    workspace.mkdir(parents=True, exist_ok=True)

    current = source
    rounds = []
    repairs = 0

    report = {
        "schema_version": 1,
        "type": "false_positive_repair_workflow",
        "original_run": str(source),
        "final_run": str(source),
        "status": "running",
        "repairs": 0,
        "max_repairs": max_repairs,
        "rounds": rounds,
    }

    report_file = workspace / "workflow.json"

    def save():
        report["final_run"] = str(current)
        report["repairs"] = repairs
        dump(report_file, report)

    save()

    while True:
        round_number = len(rounds)
        audit_root = (
            workspace / "audits" /
            f"round-{round_number:02d}"
        )

        print(
            f"\n=== AUDIT ROUND {round_number} ===",
            flush=True,
        )
        print("Source:", current, flush=True)

        try:
            status, records = audit_cycle(
                current,
                audit_root,
                model,
                confirmation_model,
                base_url,
                max_tokens,
            )
        except BaseException as exc:
            report["status"] = "error"
            report["error"] = repr(exc)
            save()
            raise

        entry = {
            "round": round_number,
            "run": str(current),
            "status": status,
            "audits": records,
        }
        rounds.append(entry)
        save()

        if status == "valid":
            report["status"] = "audited"
            save()
            return report

        if status != "false_positive":
            report["status"] = status
            save()
            return report

        if repairs >= max_repairs:
            report["status"] = "repair_limit"
            save()
            return report

        branch = (
            workspace / "branches" /
            f"repair-{repairs + 1:02d}"
        )

        print(
            "Preparing repair branch:",
            branch,
            flush=True,
        )

        try:
            origin = prepare_branch(
                current,
                branch,
                audit_root=audit_root,
            )

            if origin.get("synthetic") is not False:
                raise ValueError(
                    "Unexpected synthetic repair branch"
                )

            entry["branch"] = str(branch)
            entry["branch_origin"] = origin
            save()

            print("Resuming:", branch, flush=True)

            # The branch has its own history and manifest.
            # The original run is never resumed or modified.
            resume_simulation(branch)

            if load(branch / "manifest.json").get(
                "status"
            ) != "complete":
                raise ValueError(
                    "Repair branch did not complete"
                )

        except BaseException as exc:
            report["status"] = "repair_failed"
            report["error"] = repr(exc)
            entry["branch"] = str(branch)
            save()
            raise

        current = branch
        repairs += 1
        save()


def main():
    parser = argparse.ArgumentParser(
        description="Retrospective audit and bounded FP repair"
    )

    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v4.1-flash:thinking",
    )
    parser.add_argument("--confirm-model")
    parser.add_argument(
        "--base-url",
        default="https://nano-gpt.com/api",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    if not args.execute:
        _, _, _, candidates = inspect_run(args.run)

        print("=== DRY RUN ===")
        print("Original run:", args.run)
        print("Audit candidates:", len(candidates))

        for candidate in candidates:
            print(
                f"Turn {candidate['turn']:03d}: "
                f"{candidate['goal_id']}"
            )

        print("No network calls or writes performed.")
        return

    result = run_workflow(
        args.run,
        args.out,
        model=args.model,
        confirmation_model=args.confirm_model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        max_repairs=args.max_repairs,
    )

    print("\n=== WORKFLOW RESULT ===")
    print(json.dumps({
        "status": result["status"],
        "repairs": result["repairs"],
        "original_run": result["original_run"],
        "final_run": result["final_run"],
    }, indent=2))


if __name__ == "__main__":
    main()
