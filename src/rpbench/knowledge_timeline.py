"""Read-only, provenance-based RP knowledge timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .core import dump, load
from .observations import read_jsonl, read_observations


VERSION = "knowledge-timeline-v1"


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def make_claims(observations, attribution_events, character_ids):
    npc_text = {
        item["turn"]: item["text"]
        for item in observations
        if item["role"] == "npc"
    }

    claims = []
    seen = set()

    for index, event in enumerate(attribution_events):
        if event.get("kind") != "knowledge_claim":
            continue

        turn = event.get("turn")
        cid = event.get("character_id")
        quote = event.get("quote")
        start = event.get("start_offset")
        end = event.get("end_offset")

        if type(turn) is not int or turn not in npc_text:
            raise ValueError("Invalid knowledge-claim turn")

        if cid is not None and cid not in character_ids:
            raise ValueError("Unknown knowledge claimant")

        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Missing knowledge evidence")

        text = npc_text[turn]
        positions = []
        offset = 0

        while True:
            position = text.find(quote, offset)
            if position < 0:
                break
            positions.append(position)
            offset = position + 1

        if not positions:
            raise ValueError("Evidence not found in archived NPC text")

        if start is None or end is None:
            # Repeated quotations without unique offsets cannot
            # be used as uniquely located timeline evidence.
            if len(positions) != 1:
                raise ValueError("Ambiguous knowledge evidence span")
            start = positions[0]
            end = start + len(quote)
        elif (
            type(start) is not int
            or type(end) is not int
            or start not in positions
            or end != start + len(quote)
        ):
            raise ValueError("Knowledge evidence offset mismatch")

        key = (turn, cid, start, end)
        if key in seen:
            raise ValueError("Duplicate knowledge evidence")
        seen.add(key)

        claims.append({
            "type": "knowledge_claim_observed",
            "turn": turn,
            "role": "npc",
            "character_id": cid,
            "quote": quote,
            "start_offset": start,
            "end_offset": end,
            "attribution_event_index": index,
            "status": "candidate",
            "verified": False,
            "knowledge_acquisition_confirmed": False,
            "source_character_id": None,
            "recipient_character_id": None,
            "transfer_confirmed": False,
            "explanation": event.get("explanation", ""),
        })

    claims.sort(key=lambda e: (
        e["turn"],
        e["start_offset"],
        e["attribution_event_index"],
    ))

    for index, claim in enumerate(claims, 1):
        claim["timeline_id"] = index

    return claims


def build_timeline(run_dir, attribution_dir):
    run_dir = Path(run_dir).resolve()
    attribution_dir = Path(attribution_dir).resolve()

    source = attribution_dir / "findings.jsonl"
    attribution_manifest = attribution_dir / "manifest.json"
    run_manifest = run_dir / "manifest.json"

    if not source.is_file() or not attribution_manifest.is_file():
        raise ValueError("Missing attribution evaluation")

    manifest = load(attribution_manifest)
    identity = manifest.get("identity", {})

    from .observations import source_path

    if identity.get("source_sha256") != sha(source_path(run_dir)):
        raise ValueError("Attribution observation hash mismatch")
    if identity.get("run_manifest_sha256") != sha(run_manifest):
        raise ValueError("Attribution run manifest hash mismatch")

    # Do not import partially completed extraction runs.
    attribution_summary = load(
        attribution_dir / "summary.json"
    )
    if attribution_summary.get("status") != "complete":
        raise ValueError("Attribution evaluation is incomplete")

    characters = manifest.get("catalog", [])
    ids = {item["id"] for item in characters}
    if len(ids) != len(characters):
        raise ValueError("Duplicate character ID")

    observations = read_observations(run_dir)
    attribution_events = read_jsonl(source)

    claims = make_claims(
        observations, attribution_events, ids
    )

    provenance = {
        "version": VERSION,
        "source_run": run_dir.name,
        "attribution_evaluation": attribution_dir.name,
        "attribution_sha256": sha(source),
        "attribution_manifest_sha256": sha(attribution_manifest),
        "observations_sha256": sha(source_path(run_dir)),
    }

    timeline_id = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:16]

    target = (
        run_dir / "evaluations" /
        f"{VERSION}-{timeline_id}"
    )
    if target.exists():
        raise FileExistsError(
            f"Timeline already exists: {target}"
        )

    target.mkdir(parents=True, exist_ok=False)

    dump(target / "manifest.json", {
        "schema_version": 1,
        "provenance": provenance,
        "method": "Read-only observed knowledge claims",
        "confirmed_state_mutation": False,
    })

    with (target / "events.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for claim in claims:
            f.write(json.dumps(
                claim, ensure_ascii=False
            ) + "\n")

    summary = {
        "status": "complete",
        "observed_claims": len(claims),
        "identified_claimants": sum(
            e["character_id"] is not None for e in claims
        ),
        "unknown_claimants": sum(
            e["character_id"] is None for e in claims
        ),
        "confirmed_transfers": 0,
        "confirmed_acquisitions": 0,
        "score": None,
    }
    dump(target / "summary.json", summary)

    return target, summary


def main():
    parser = argparse.ArgumentParser(
        description="Build read-only RP knowledge timeline"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--attribution", required=True)
    args = parser.parse_args()

    target, summary = build_timeline(
        args.run, args.attribution
    )
    print(target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
