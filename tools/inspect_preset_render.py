"""Inspect a SillyTavern preset against one refusal-pilot scenario."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rpbench.preset_adapter import (
    SillyTavernPresetAdapter,
    render_with_adapter,
)
from rpbench.prompt_models import PromptMessage
from rpbench.refusal_probe import (
    build_messages,
    validate_scenario,
)


MACRO_PATTERN = re.compile(r"\{\{[^{}\n]+\}\}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--character-id", type=int, default=100001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    destination = Path(args.out)
    if destination.exists():
        parser.error(f"Output exists: {destination}")

    scenario = json.loads(
        Path(args.scenario).read_text(encoding="utf-8")
    )
    validate_scenario(scenario)

    baseline = build_messages(scenario)

    # Keep the final player turn in the history.
    # Character/system information is provided through the
    # SillyTavern marker rather than duplicated as a system turn.
    history = [
        PromptMessage(m["role"], m["content"])
        for m in baseline[1:]
    ]

    adapter = SillyTavernPresetAdapter(
        args.preset,
        preset_id=Path(args.preset).stem,
        allow_disabled_orphans=True,
    )

    ordered = adapter.preset.ordered_prompts(args.character_id)
    active_markers = {
        p["identifier"]
        for p in ordered
        if p["enabled"] and p.get("marker", False)
    }

    # These are explicit placeholders for the pilot.
    # Empty values mean the pilot does not supply that ST field.
    marker_content = {
        name: "" for name in active_markers
        if name != "chatHistory"
    }

    if "charDescription" in marker_content:
        marker_content["charDescription"] = scenario["system"]
    elif "main" in marker_content:
        marker_content["main"] = scenario["system"]
    else:
        raise ValueError(
            "No charDescription marker; cannot place scenario system"
        )

    result = render_with_adapter(
        adapter,
        history,
        markers=marker_content,
        character_name="Mara Voss",
        user_name="Player",
        character_id=args.character_id,
    )

    # Presets may append post-history system messages.
    # Verify that every history message remains intact and ordered,
    # rather than requiring the final message to be the player.
    original_history = [
        message.to_dict() for message in history
    ]

    cursor = 0
    history_positions = []

    for index, message in enumerate(result.messages):
        if (
            cursor < len(original_history)
            and message == original_history[cursor]
        ):
            history_positions.append(index)
            cursor += 1

    if cursor != len(original_history):
        raise ValueError(
            "Original fixed history is missing or reordered: "
            f"matched {cursor}/{len(original_history)} messages"
        )

    rendered_user_turns = [
        message["content"]
        for message in result.messages
        if message["role"] == "user"
    ]

    original_user_turns = [
        message.content
        for message in history
        if message.role == "user"
    ]

    if rendered_user_turns != original_user_turns:
        raise ValueError("Player history was changed")

    trailing_messages = result.messages[
        history_positions[-1] + 1:
    ]

    unresolved = sorted({
        macro
        for message in result.messages
        for macro in MACRO_PATTERN.findall(message["content"])
    })

    # Explicit diagnostics only. Unknown ST macros may exist
    # outside the simple {{...}} syntax as well.
    data = {
        "preset_id": result.preset_id,
        "preset_sha256": result.source_sha256,
        "scenario_id": scenario["id"],
        "character_id": args.character_id,
        "applied_hooks": result.applied_hooks,
        "active_prompt_count": sum(
            p["enabled"] for p in ordered
        ),
        "active_nonempty_prompt_count": sum(
            p["enabled"] and bool(p.get("content", "").strip())
            for p in ordered
        ),
        "message_count": len(result.messages),
        "message_roles": [m["role"] for m in result.messages],
        "total_content_characters": sum(
            len(m["content"]) for m in result.messages
        ),
        "unresolved_macros": unresolved,
        "empty_markers": sorted(
            name for name, content in marker_content.items()
            if not content
        ),
        "baseline_messages_sha256": hashlib.sha256(
            json.dumps(
                baseline,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "rendered_messages_sha256": hashlib.sha256(
            json.dumps(
                result.messages,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)

    (destination / "messages.json").write_text(
        json.dumps(
            result.messages,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    (destination / "inspection.json").write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("\nSaved:", destination)

    if unresolved:
        print("\nWARNING: unresolved SillyTavern macros")
        print("Rendering is not yet verified for inference.")


if __name__ == "__main__":
    main()
