"""Independent fixed-context NPC comparisons for the existing blind A/B reviewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .characters import build_npc_prompt
from .core import dump, load
from .sampling import validate_sampling
from .simulation import (
    apply_prompt_profile,
    create_player_prompt,
    make_backend,
)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def read_cases(path):
    path = Path(path).resolve()
    data = load(path)

    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported NPC A/B case set")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected nonempty cases list")

    seen = set()

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Invalid case entry")

        cid = case.get("id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Invalid or duplicate case ID")

        seen.add(cid)

        if case.get("language") not in {"en", "de"}:
            raise ValueError(f"{cid}: invalid language")

        if not isinstance(case.get("scenario"), str):
            raise ValueError(f"{cid}: missing scenario")

        for key in ("context", "player"):
            if (
                not isinstance(case.get(key), str)
                or not case[key].strip()
            ):
                raise ValueError(f"{cid}: missing {key}")

        history = case.get("history", [])

        if (
            not isinstance(history, list)
            or len(history) % 2 != 0
            or any(
                not isinstance(message, dict)
                or message.get("role") != (
                    "user" if index % 2 == 0 else "assistant"
                )
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for index, message in enumerate(history)
            )
        ):
            raise ValueError(
                f"{cid}: history must contain complete user/assistant pairs"
            )

        scenario_path = (
            path.parent / case["scenario"]
        ).resolve()

        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)

    return path, cases


def build_case_messages(case, cases_path, config_path, config):
    scenario_path = (
        cases_path.parent / case["scenario"]
    ).resolve()

    scenario = load(scenario_path)

    npc_prompt, _ = build_npc_prompt(
        scenario,
        scenario_path,
    )

    player_path = scenario_path.parent / "player.json"
    player_prompt = create_player_prompt(load(player_path))

    _, npc_prompt, _ = apply_prompt_profile(
        player_prompt,
        npc_prompt,
        config,
        config_path,
    )

    # The entire context is fixed in the case definition.
    # Outputs from other test cases never enter this history.
    return [
        {"role": "system", "content": npc_prompt},
        {"role": "user", "content": case["context"]},
        *case.get("history", []),
        {"role": "user", "content": case["player"]},
    ]


def build_specs(cases_path, cases, config_path, config):
    backend = make_backend(config)

    if backend.mode != "chat":
        raise ValueError(
            "NPC A/B v1 supports chat completion only"
        )

    sampling = config.get("sampling", {})
    validate_sampling(
        backend.name,
        backend.mode,
        sampling,
    )

    specs = []

    for index, case in enumerate(cases, 1):
        messages = build_case_messages(
            case,
            cases_path,
            config_path,
            config,
        )

        endpoint, payload = backend.prepare(
            messages,
            sampling,
            config.get("prompt_format", {}),
        )

        specs.append({
            "index": index,
            "id": case["id"],
            "language": case["language"],
            "player": case["player"],
            "messages_sha256": digest(messages),
            "endpoint": endpoint,
            "payload": payload,
        })

    return specs


def run(cases_file, config_file, output):
    cases_path, cases = read_cases(cases_file)

    config_path = Path(config_file).resolve()
    config = load(config_path)

    specs = build_specs(
        cases_path,
        cases,
        config_path,
        config,
    )

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"

    identity = {
        "schema_version": 1,
        "type": "fixed_context_npc_ab",
        "case_set_sha256": hashlib.sha256(
            cases_path.read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "model": config["backend"]["model"],
        "cases": [
            {
                "id": spec["id"],
                "messages_sha256": spec["messages_sha256"],
            }
            for spec in specs
        ],
    }

    if manifest_path.exists():
        saved = load(manifest_path)

        if any(
            saved.get(key) != value
            for key, value in identity.items()
        ):
            raise ValueError(
                "Existing run has different configuration, "
                "cases or input hashes"
            )
    else:
        dump(
            manifest_path,
            {
                **identity,
                "status": "running",
                "completed_turns": 0,
            },
        )

    backend = make_backend(config)

    cases_dir = root / "cases"
    cases_dir.mkdir(exist_ok=True)

    for spec in specs:
        result_path = (
            cases_dir / f'{spec["index"]:03d}.json'
        )

        if result_path.exists():
            existing = load(result_path)

            if existing.get("request_sha256") != digest(
                spec["payload"]
            ):
                raise ValueError(
                    f'Case {spec["id"]}: saved request mismatch'
                )

            if (
                not isinstance(existing.get("answer"), str)
                or not existing["answer"].strip()
            ):
                raise ValueError(
                    f'Case {spec["id"]}: saved answer invalid'
                )

            continue

        answer, raw = backend.generate(
            spec["endpoint"],
            spec["payload"],
        )

        choice = (
            (raw.get("choices") or [{}])[0]
            if isinstance(raw, dict)
            else {}
        )

        finish = choice.get("finish_reason")

        if finish == "length" or not answer.strip():
            raise RuntimeError(
                f'Case {spec["id"]}: truncated or empty NPC answer'
            )

        dump(
            result_path,
            {
                "id": spec["id"],
                "request_sha256": digest(spec["payload"]),
                "input_sha256": spec["messages_sha256"],
                "endpoint": spec["endpoint"],
                "payload": spec["payload"],
                "answer": answer,
                "raw_response": raw,
            },
        )

        print(
            f'Completed {spec["index"]}/{len(specs)}: '
            f'{spec["id"]}',
            flush=True,
        )

    # Publish observations only after every case succeeds.
    observations = []

    for spec in specs:
        result = load(
            cases_dir / f'{spec["index"]:03d}.json'
        )

        observations.extend([
            {
                "type": "dialogue_observation",
                "turn": spec["index"],
                "role": "player",
                "text": spec["player"],
            },
            {
                "type": "dialogue_observation",
                "turn": spec["index"],
                "role": "npc",
                "text": result["answer"],
            },
        ])

    target = root / "observations.jsonl"
    temporary = root / ".observations.jsonl.tmp"

    temporary.write_text(
        "".join(
            canonical(entry) + "\n"
            for entry in observations
        ),
        encoding="utf-8",
    )

    temporary.replace(target)

    dump(
        manifest_path,
        {
            **identity,
            "status": "complete",
            "completed_turns": len(specs),
        },
    )

    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--cases", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    print(run(args.cases, args.config, args.out))


if __name__ == "__main__":
    main()
