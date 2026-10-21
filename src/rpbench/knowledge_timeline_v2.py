"""Read-only timeline combining knowledge claims and transfer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .core import dump, load
from .knowledge_timeline import make_claims, sha
from .observations import (
    read_jsonl,
    read_observations,
    source_path,
)


VERSION = "knowledge-timeline-v2"
KINDS = {"explicit_transfer", "reported_transfer"}
PRIOR = {"known", "unknown", "not_known"}


def digest(value):
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def validate_span(text, event):
    quote = event.get("quote")
    start = event.get("start_offset")
    end = event.get("end_offset")

    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Missing timeline evidence")

    positions = []
    offset = 0
    while True:
        pos = text.find(quote, offset)
        if pos < 0:
            break
        positions.append(pos)
        offset = pos + 1

    if not positions:
        raise ValueError("Timeline evidence not found in NPC text")

    if start is None or end is None:
        if len(positions) != 1:
            raise ValueError("Ambiguous timeline evidence span")
        start = positions[0]
        end = start + len(quote)
    elif (
        type(start) is not int
        or type(end) is not int
        or start not in positions
        or end != start + len(quote)
    ):
        raise ValueError("Timeline evidence offset mismatch")

    return start, end


def make_transfers(observations, transfer_events, character_ids):
    npc_text = {
        item["turn"]: item["text"]
        for item in observations
        if item["role"] == "npc"
    }

    result = []
    seen = set()

    for index, event in enumerate(transfer_events):
        turn = event.get("turn")
        kind = event.get("kind")
        source = event.get("source_character_id")
        recipient = event.get("recipient_character_id")
        fact = event.get("fact")
        prior = event.get("recipient_prior_knowledge")

        if type(turn) is not int or turn not in npc_text:
            raise ValueError("Invalid transfer turn")
        if event.get("role") != "npc":
            raise ValueError("Transfer must reference NPC output")
        if kind not in KINDS:
            raise ValueError("Invalid transfer kind")
        if source is not None and source not in character_ids:
            raise ValueError("Unknown transfer source")
        if recipient is not None and recipient not in character_ids:
            raise ValueError("Unknown transfer recipient")
        if source is not None and source == recipient:
            raise ValueError("Transfer source equals recipient")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("Missing transfer fact")
        if prior not in PRIOR:
            raise ValueError("Invalid prior knowledge")

        # A judge result is NEVER trusted as confirmed knowledge.
        if (
            event.get("verified") is not False
            or event.get("acquisition_confirmed") is not False
            or event.get("status") != "candidate"
        ):
            raise ValueError("Unexpected confirmed transfer candidate")

        start, end = validate_span(npc_text[turn], event)

        key = (turn, kind, source, recipient, fact, start, end)
        if key in seen:
            raise ValueError("Duplicate transfer evidence")
        seen.add(key)

        result.append({
            "type": "knowledge_transfer_candidate",
            "turn": turn,
            "role": "npc",
            "kind": kind,
            "source_character_id": source,
            "recipient_character_id": recipient,
            "fact": fact,
            "recipient_prior_knowledge": prior,
            "quote": event["quote"],
            "start_offset": start,
            "end_offset": end,
            "transfer_event_index": index,
            "explanation": event.get("explanation", ""),
            "status": "candidate",
            "verified": False,
            "transfer_confirmed": False,
            "knowledge_acquisition_confirmed": False,
        })

    return result


def merge_events(claims, transfers):
    merged = []
    for event in claims:
        item = dict(event)
        item.pop("timeline_id", None)
        item["evidence_source"] = "character_attribution"
        merged.append(item)

    for event in transfers:
        item = dict(event)
        item["evidence_source"] = "knowledge_transfers"
        merged.append(item)

    merged.sort(key=lambda event: (
        event["turn"],
        event["start_offset"],
        0 if event["type"] == "knowledge_claim_observed" else 1,
        event.get(
            "attribution_event_index",
            event.get("transfer_event_index", 0),
        ),
    ))

    for index, event in enumerate(merged, 1):
        event["timeline_id"] = index

    return merged


def validate_evaluation(run_dir, evaluation_dir, expected_kind):
    manifest_path = evaluation_dir / "manifest.json"
    summary_path = evaluation_dir / "summary.json"
    findings_path = evaluation_dir / "findings.jsonl"

    if not all(path.is_file() for path in (
        manifest_path, summary_path, findings_path
    )):
        raise ValueError(f"Incomplete {expected_kind} files")

    manifest = load(manifest_path)
    summary = load(summary_path)
    identity = manifest.get("identity", {})

    if summary.get("status") != "complete":
        raise ValueError(f"{expected_kind} evaluation is incomplete")

    if expected_kind == "attribution":
        source_key = "source_sha256"
        manifest_key = "run_manifest_sha256"
        version = "character-attribution-v2"
    else:
        source_key = "source_hash"
        manifest_key = "manifest_hash"
        version = "knowledge-transfers-v2"

    if identity.get("version") != version:
        raise ValueError(f"{expected_kind} version mismatch")
    if identity.get(source_key) != sha(source_path(run_dir)):
        raise ValueError(f"{expected_kind} observation hash mismatch")
    if identity.get(manifest_key) != sha(run_dir / "manifest.json"):
        raise ValueError(f"{expected_kind} run manifest hash mismatch")

    catalog = manifest.get("catalog")
    if not isinstance(catalog, list):
        raise ValueError(f"Missing {expected_kind} character catalog")

    ids = [entry["id"] for entry in catalog]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate character IDs")

    return manifest, read_jsonl(findings_path)


def build_timeline(run_dir, attribution_dir, transfer_dir):
    run_dir = Path(run_dir).resolve()
    attribution_dir = Path(attribution_dir).resolve()
    transfer_dir = Path(transfer_dir).resolve()

    attribution_manifest, attribution_events = validate_evaluation(
        run_dir, attribution_dir, "attribution"
    )
    transfer_manifest, transfer_events = validate_evaluation(
        run_dir, transfer_dir, "transfers"
    )

    attribution_catalog = attribution_manifest["catalog"]
    transfer_catalog = transfer_manifest["catalog"]

    attribution_ids = {entry["id"] for entry in attribution_catalog}
    transfer_ids = {entry["id"] for entry in transfer_catalog}

    if attribution_ids != transfer_ids:
        raise ValueError("Character catalogs disagree")

    observations = read_observations(run_dir)

    claims = make_claims(
        observations, attribution_events, attribution_ids
    )
    transfers = make_transfers(
        observations, transfer_events, attribution_ids
    )
    events = merge_events(claims, transfers)

    provenance = {
        "version": VERSION,
        "source_run": run_dir.name,
        "observations_sha256": sha(source_path(run_dir)),
        "run_manifest_sha256": sha(run_dir / "manifest.json"),
        "attribution_evaluation": attribution_dir.name,
        "attribution_manifest_sha256": sha(
            attribution_dir / "manifest.json"
        ),
        "attribution_findings_sha256": sha(
            attribution_dir / "findings.jsonl"
        ),
        "transfer_evaluation": transfer_dir.name,
        "transfer_manifest_sha256": sha(
            transfer_dir / "manifest.json"
        ),
        "transfer_findings_sha256": sha(
            transfer_dir / "findings.jsonl"
        ),
    }

    target = (
        run_dir / "evaluations" /
        f"{VERSION}-{digest(provenance)}"
    )

    if target.exists():
        raise FileExistsError(
            f"Timeline already exists: {target}"
        )

    target.mkdir(parents=True, exist_ok=False)

    dump(target / "manifest.json", {
        "schema_version": 2,
        "provenance": provenance,
        "method": "Read-only chronological candidate evidence",
        "confirmed_state_mutation": False,
    })

    with (target / "events.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for event in events:
            output.write(
                json.dumps(event, ensure_ascii=False) + "\n"
            )

    summary = {
        "status": "complete",
        "total_events": len(events),
        "observed_claims": len(claims),
        "transfer_candidates": len(transfers),
        "explicit_candidates": sum(
            t["kind"] == "explicit_transfer" for t in transfers
        ),
        "reported_candidates": sum(
            t["kind"] == "reported_transfer" for t in transfers
        ),
        "confirmed_transfers": 0,
        "confirmed_acquisitions": 0,
        "score": None,
    }
    dump(target / "summary.json", summary)

    return target, summary


def main():
    parser = argparse.ArgumentParser(
        description="Build read-only RP knowledge timeline v2"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--transfers", required=True)
    args = parser.parse_args()

    target, summary = build_timeline(
        args.run, args.attribution, args.transfers
    )
    print(target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
