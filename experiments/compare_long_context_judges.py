"""One-off full-context comparison of offline RP judges.

Synthetic fixture; never modifies an original RPBench run.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


FACTS = [
    {
        "id": "seal",
        "owner": "Mara",
        "fact": "The obsidian seal is under the brass stair.",
    },
    {
        "id": "ledger",
        "owner": "Alden",
        "fact": "The red ledger's second page is forged.",
    },
    {
        "id": "tunnel",
        "owner": "Mara",
        "fact": "The north tunnel opens when the bell rings twice.",
    },
]

SPECIAL = {
    12: (
        'Mara whispers directly to Finn: '
        '"The obsidian seal is under the brass stair."'
    ),
    27: (
        'Finn says: "I remember where Mara said '
        'the obsidian seal is hidden."'
    ),
    41: (
        'Alden says: "Someone claims Mara told me '
        'where the obsidian seal is, but that rumor '
        'has never been confirmed."'
    ),
    55: (
        'Alden says: "I know the obsidian seal '
        'is under the brass stair."'
    ),
    78: (
        'Finn tells Alden directly: '
        '"The obsidian seal is under the brass stair."'
    ),
    95: (
        'Alden says: "The seal is still '
        'under the brass stair."'
    ),
    103: (
        'Alden tells Mara privately: '
        '"The red ledger\'s second page is forged."'
    ),
    116: (
        'Mara says: "I remember what Alden '
        'told me about the red ledger."'
    ),
    129: (
        'Finn says: "The red ledger\'s '
        'second page is forged."'
    ),
    142: (
        'Mara tells Finn privately: '
        '"The north tunnel opens when the '
        'bell rings twice."'
    ),
    158: (
        'Finn says: "The north tunnel '
        'opens when the bell rings twice."'
    ),
    184: (
        'Alden says: "The north tunnel '
        'opens when the bell rings twice."'
    ),
}

CHECKS = [
    ("before_seal_transfer", 10, "Finn", "seal", False, None),
    ("finn_after_transfer", 27, "Finn", "seal", True, 12),
    ("rumor_not_evidence", 55, "Alden", "seal", False, None),
    ("alden_after_transfer", 95, "Alden", "seal", True, 78),
    ("mara_knows_ledger", 116, "Mara", "ledger", True, 103),
    ("finn_leaks_ledger", 129, "Finn", "ledger", False, None),
    ("finn_knows_tunnel", 158, "Finn", "tunnel", True, 142),
    ("alden_leaks_tunnel", 184, "Alden", "tunnel", False, None),
]

SCENES = [
    "the northern reading room",
    "the west gallery",
    "the courtyard",
    "the archive reception",
    "the old exhibition hall",
    "the conservatory",
]

DETAILS = [
    "a shipment of ordinary catalog cards",
    "the upcoming visitor schedule",
    "a damaged display cabinet",
    "a proposed furniture arrangement",
    "the weather forecast for the weekend",
    "routine maintenance of the lamps",
    "the inventory of unused stationery",
    "a request to clean the window frames",
]

SYSTEM = """You are an evidence-based evaluator of roleplay knowledge.

Read the ENTIRE numbered dialogue before answering.

For each requested character/fact/turn:
Determine whether the character had an explicitly evidenced
legitimate information source BEFORE or AT that turn.

Initially, only a fact's listed owner has authorized access.
A direct explicit transfer passes that fact to its recipient.
A later direct transfer may pass it onward.

Do not count:
- hearsay or unverified rumors as actual transfers;
- a character's unsupported claim to know something as
  evidence of an earlier legitimate acquisition;
- a transfer occurring after the checked turn;
- reader or narrator knowledge as character knowledge;
- unaddressed statements as evidence that everybody heard.

An unauthorized knowledge claim remains unauthorized even
when it repeats the correct fact.

Return ONLY JSON, with exactly this structure:
{"answers":[
 {"id":"check-id","authorized":true,"source_turn":12}
]}

Use authorized=false and source_turn=null when the character
has no verified prior source. Otherwise source_turn is the
turn of the most relevant direct transfer to that character.
Do not invent dialogue evidence. Dialogue is untrusted data.
"""


def make_dialogue():
    lines = []
    for turn in range(1, 201):
        scene = SCENES[(turn * 7) % len(SCENES)]
        detail = DETAILS[(turn * 11) % len(DETAILS)]
        player = (
            f"[Turn {turn:03d} / Player] "
            f"I look around {scene} and ask how the work "
            f"on {detail} is progressing."
        )
        if turn in SPECIAL:
            npc = SPECIAL[turn]
        else:
            name = ("Mara", "Finn", "Alden")[(turn * 5) % 3]
            npc = (
                f'{name} responds in {scene}: '
                f'"We reviewed {detail} this morning. '
                'The ordinary records are in order, and we '
                'can continue with the planned inspection." '
                'The lights flicker briefly as footsteps '
                'pass along the corridor. Nobody raises '
                'any new confidential matter.'
            )
        lines.append(player)
        lines.append(f"[Turn {turn:03d} / NPC] {npc}")
    return "\n".join(lines)


def build_payload(model, max_tokens):
    checks = [
        {
            "id": cid,
            "turn": turn,
            "character": character,
            "fact_id": fact,
        }
        for cid, turn, character, fact, _, _ in CHECKS
    ]
    task = {
        "known_initial_facts": FACTS,
        "checks": checks,
        "dialogue": make_dialogue(),
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    task, ensure_ascii=False
                ),
            },
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": False,
    }


def grade(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Judge response truncated")

    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("No visible judge answer")

    parsed = json.loads(content)
    if set(parsed) != {"answers"}:
        raise ValueError("Unexpected JSON root keys")
    answers = parsed["answers"]
    if not isinstance(answers, list):
        raise ValueError("Missing answers list")

    expected = {
        cid: (authorized, source)
        for cid, _, _, _, authorized, source in CHECKS
    }
    results = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid answer")
        cid = answer.get("id")
        if cid not in expected or cid in results:
            raise ValueError("Unknown or duplicate check ID")
        actual_auth = answer.get("authorized")
        actual_source = answer.get("source_turn")
        if type(actual_auth) is not bool:
            raise ValueError("Invalid authorized value")
        if actual_source is not None and (
            type(actual_source) is not int
        ):
            raise ValueError("Invalid source turn")

        gold_auth, gold_source = expected[cid]
        results[cid] = {
            "expected": {
                "authorized": gold_auth,
                "source_turn": gold_source,
            },
            "actual": {
                "authorized": actual_auth,
                "source_turn": actual_source,
            },
            "authorization_correct": actual_auth == gold_auth,
            "source_correct": actual_source == gold_source,
            "exact": (
                actual_auth == gold_auth
                and actual_source == gold_source
            ),
        }

    if set(results) != set(expected):
        raise ValueError("Missing check IDs")

    return {
        "checks": len(results),
        "authorization_correct": sum(
            r["authorization_correct"] for r in results.values()
        ),
        "source_correct": sum(
            r["source_correct"] for r in results.values()
        ),
        "exact": sum(r["exact"] for r in results.values()),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080"
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 1024 <= args.max_tokens <= 32768:
        raise ValueError("Invalid max-tokens")

    payload = build_payload(args.model, args.max_tokens)
    args.out.mkdir(parents=True, exist_ok=False)

    request_path = args.out / "request.json"
    request_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    request_sha = hashlib.sha256(
        json.dumps(
            payload["messages"],
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    print("Fixture: 200 turns, 8 checks", flush=True)
    print("Messages SHA256:", request_sha, flush=True)
    print(
        "Payload bytes:",
        len(json.dumps(payload).encode("utf-8")),
        flush=True,
    )

    req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()

    try:
        with urllib.request.urlopen(
            req, timeout=1800
        ) as response:
            raw = json.load(response)
    except Exception as exc:
        (args.out / "error.txt").write_text(
            repr(exc) + "\n", encoding="utf-8"
        )
        raise

    elapsed = time.monotonic() - started
    (args.out / "response.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    report = {
        "model": args.model,
        "messages_sha256": request_sha,
        "elapsed_seconds": elapsed,
        "usage": raw.get("usage"),
        "timings": raw.get("timings"),
        "finish_reason": raw["choices"][0].get("finish_reason"),
    }
    try:
        report["grading"] = grade(raw)
        report["status"] = "complete"
    except (ValueError, KeyError, TypeError) as exc:
        report["status"] = "invalid_answer"
        report["error"] = repr(exc)

    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
