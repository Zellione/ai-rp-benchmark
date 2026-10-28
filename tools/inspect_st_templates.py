"""Inspect active SillyTavern template constructs without executing them."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from rpbench.prompt_presets import SillyTavernChatPreset


MACRO = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


def classify(expression):
    value = expression.strip()

    if re.match(r"^if\s+", value):
        return "if"
    if value == "else":
        return "else"
    if value == "/if":
        return "endif"
    if re.match(r"^\.[\w.]+\s*=", value):
        return "assignment"
    if value.lower() == "trim":
        return "trim"
    if value.lower() in {"user", "char"}:
        return "name"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--character-id", type=int, default=100001)
    args = parser.parse_args()

    source = Path(args.preset)

    preset = SillyTavernChatPreset.from_file(
        source,
        allow_disabled_orphans=True,
    )
    ordered = preset.ordered_prompts(args.character_id)

    counts = collections.Counter()
    unknown = collections.Counter()
    assignments = collections.Counter()
    blocks = []

    for prompt in ordered:
        if not prompt["enabled"]:
            continue

        content = prompt.get("content", "")
        expressions = MACRO.findall(content)

        if not content.strip():
            continue

        categories = collections.Counter(
            classify(x) for x in expressions
        )
        counts.update(categories)

        for expression in expressions:
            category = classify(expression)
            value = expression.strip()

            if category == "unknown":
                unknown[value] += 1
            elif category == "assignment":
                assignments[value] += 1

        blocks.append({
            "id": prompt["identifier"],
            "name": prompt.get("name"),
            "characters": len(content),
            "macros": len(expressions),
            "categories": dict(categories),
        })

    print("=== PRESET ===")
    print("File:", source)
    print("SHA256:", hashlib.sha256(source.read_bytes()).hexdigest())
    print("Character ID:", args.character_id)

    print("\n=== MACRO CATEGORIES ===")
    for name, count in counts.most_common():
        print(f"{name:14} {count}")

    print("\n=== VARIABLE ASSIGNMENTS ===")
    for expression, count in assignments.most_common():
        print(f"{count:3}  {expression}")

    print("\n=== UNKNOWN CONSTRUCTS ===")
    if unknown:
        for expression, count in unknown.most_common():
            print(f"{count:3}  {expression}")
    else:
        print("None")

    print("\n=== ACTIVE TEXT BLOCKS ===")
    for block in blocks:
        print(
            f"{block['characters']:5} chars | "
            f"{block['macros']:3} macros | "
            f"{block['name']}"
        )

    print("\n=== CONTROL FLOW BLOCKS ===")
    for block in blocks:
        categories = block["categories"]
        if any(categories.get(k, 0) for k in ("if", "else", "endif")):
            print(
                f"{block['name']}: "
                f"{json.dumps(categories, ensure_ascii=False)}"
            )

    print("\n=== STRUCTURAL WARNINGS ===")
    for block in blocks:
        name = (block["name"] or "").lower()
        if "readme" in name or "documentation" in name:
            print("Documentation-like active block:", block["name"])


if __name__ == "__main__":
    main()
