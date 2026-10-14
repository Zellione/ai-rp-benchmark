"""Evidence-based, append-only human review for RPBench."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .core import append, dump, load
from .state import (
    VALID_EVENT_TYPES,
    apply_confirmed_event,
    initial_state,
)

EVALUATION_TYPES = {
    "persona_violation",
    "lore_error",
    "refusal",
    "agency_violation",
}

ALL_TYPES = VALID_EVENT_TYPES | EVALUATION_TYPES


def now():
    return datetime.now(timezone.utc).isoformat()


def records(path):
    path = Path(path)
    if not path.exists():
        return []

    result = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL: {path}:{number}"
            ) from exc
    return result


def observations(run_dir):
    return records(Path(run_dir) / "observations.jsonl")


def proposals(run_dir):
    return records(Path(run_dir) / "review_proposals.jsonl")


def decisions(run_dir):
    return records(Path(run_dir) / "review_decisions.jsonl")


def find_evidence(run_dir, turn, role, quote):
    if type(turn) is not int or turn < 1:
        raise ValueError("Invalid turn")
    if role not in {"player", "npc"}:
        raise ValueError("Invalid role")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("A nonempty evidence quote is required")

    matches = [
        o for o in observations(run_dir)
        if o["turn"] == turn and o["role"] == role
        and quote in o["text"]
    ]

    if len(matches) != 1:
        raise ValueError(
            "Evidence must match exactly one archived observation"
        )

    return matches[0]


def validate_payload(kind, payload):
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    required = {
        "knowledge_acquired": ("character_id", "fact"),
        "relationship_changed": (
            "character_id", "target", "description"
        ),
        "goal_status_changed": ("goal_id", "status"),
    }.get(kind, ())

    for field in required:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid field: {field}")

    if kind == "goal_status_changed":
        if payload["status"] not in {
            "pending", "active", "completed", "failed"
        }:
            raise ValueError("Invalid goal status")

    if kind in EVALUATION_TYPES and payload:
        raise ValueError(
            "Evaluation events do not accept state-change payloads"
        )


def propose(
    run_dir, kind, turn, role, quote,
    rationale, payload=None,
):
    run_dir = Path(run_dir)
    if kind not in ALL_TYPES:
        raise ValueError(f"Unknown review type: {kind}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Review rationale required")

    payload = {} if payload is None else payload
    validate_payload(kind, payload)
    find_evidence(run_dir, turn, role, quote)

    existing = proposals(run_dir)
    proposal_id = len(existing) + 1

    entry = {
        "id": proposal_id,
        "type": kind,
        "turn": turn,
        "role": role,
        "quote": quote,
        "rationale": rationale,
        "payload": payload,
        "created_utc": now(),
    }

    append(run_dir / "review_proposals.jsonl", entry)
    return entry


def rebuild(run_dir, additional_decision=None):
    """Replay confirmed state proposals without trusting snapshots."""
    run_dir = Path(run_dir)
    truth = load(run_dir / "ground_truth.json")
    state = initial_state(truth)

    all_proposals = proposals(run_dir)
    all_decisions = decisions(run_dir)
    if additional_decision is not None:
        all_decisions.append(additional_decision)
    indexed = {p["id"]: p for p in all_proposals}

    if len(indexed) != len(all_proposals):
        raise ValueError("Duplicate proposal ID")

    seen = set()
    evaluations = []

    for decision in all_decisions:
        pid = decision["proposal_id"]

        if pid not in indexed or pid in seen:
            raise ValueError("Unknown or duplicate review decision")
        seen.add(pid)

        if decision["decision"] != "accept":
            continue

        proposal = indexed[pid]
        find_evidence(
            run_dir,
            proposal["turn"],
            proposal["role"],
            proposal["quote"],
        )

        if proposal["type"] in EVALUATION_TYPES:
            evaluations.append({
                "proposal_id": pid,
                "type": proposal["type"],
                "turn": proposal["turn"],
                "role": proposal["role"],
                "quote": proposal["quote"],
                "rationale": proposal["rationale"],
            })
            continue

        event = {
            **proposal["payload"],
            "event_id": state["last_confirmed_event"] + 1,
            "type": proposal["type"],
            "verified": True,
            "verified_by": "human",
            "evidence_proposal_id": pid,
            "evidence_turn": proposal["turn"],
        }
        state = apply_confirmed_event(state, event)

    return state, evaluations


def atomic_dump(path, value):
    path = Path(path)
    data = json.dumps(
        value, ensure_ascii=False, indent=2
    ) + "\n"

    fd, temporary = tempfile.mkstemp(
        prefix=".review-", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_reviewed_state(run_dir):
    state, evaluations = rebuild(run_dir)
    atomic_dump(
        Path(run_dir) / "reviewed_state.json", state
    )
    atomic_dump(
        Path(run_dir) / "review_evaluations.json",
        evaluations,
    )
    return state, evaluations


def decide(run_dir, proposal_id, decision, reviewer, note=""):
    run_dir = Path(run_dir)

    if decision not in {"accept", "reject"}:
        raise ValueError("Decision must be accept or reject")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Reviewer identifier required")
    if type(proposal_id) is not int:
        raise ValueError("Invalid proposal ID")

    indexed = {p["id"]: p for p in proposals(run_dir)}
    if proposal_id not in indexed:
        raise ValueError("Unknown proposal ID")
    if any(
        d["proposal_id"] == proposal_id
        for d in decisions(run_dir)
    ):
        raise ValueError("Proposal already decided")

    entry = {
        "proposal_id": proposal_id,
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "decided_utc": now(),
    }

    # Validate the entire candidate replay before journaling.
    rebuild(run_dir, additional_decision=entry)
    append(run_dir / "review_decisions.jsonl", entry)
    save_reviewed_state(run_dir)

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Evidence-based RPBench review"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose")
    p.add_argument("--run", required=True)
    p.add_argument("--type", required=True, choices=sorted(ALL_TYPES))
    p.add_argument("--turn", required=True, type=int)
    p.add_argument("--role", required=True, choices=["player", "npc"])
    p.add_argument("--quote", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--payload", default="{}")

    d = sub.add_parser("decide")
    d.add_argument("--run", required=True)
    d.add_argument("--id", required=True, type=int)
    d.add_argument("--decision", required=True, choices=["accept", "reject"])
    d.add_argument("--reviewer", required=True)
    d.add_argument("--note", default="")

    r = sub.add_parser("rebuild")
    r.add_argument("--run", required=True)

    l = sub.add_parser("list")
    l.add_argument("--run", required=True)

    args = parser.parse_args()

    if args.command == "propose":
        result = propose(
            args.run, args.type, args.turn, args.role,
            args.quote, args.rationale,
            json.loads(args.payload),
        )
    elif args.command == "decide":
        result = decide(
            args.run, args.id, args.decision,
            args.reviewer, args.note,
        )
    elif args.command == "rebuild":
        state, evaluations = save_reviewed_state(args.run)
        result = {
            "confirmed_state_events":
                state["last_confirmed_event"],
            "evaluation_count": len(evaluations),
        }
    else:
        completed = {
            d["proposal_id"]: d
            for d in decisions(args.run)
        }
        result = [
            {
                **p,
                "decision": completed.get(p["id"]),
            }
            for p in proposals(args.run)
        ]

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
