"""Structured character cards and auditable NPC prompt composition."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED = (
    "id", "name", "appearance", "personality",
    "speech", "knowledge", "private_knowledge",
    "invariants",
)


def load_character_card(path: str | Path) -> dict:
    path = Path(path)
    card = json.loads(path.read_text(encoding="utf-8"))

    for field in REQUIRED:
        if field not in card:
            raise ValueError(f"{path}: missing field {field}")

    for field in (
        "appearance", "personality", "knowledge",
        "private_knowledge", "invariants",
    ):
        if not isinstance(card[field], list):
            raise ValueError(f"{path}: {field} must be a list")

    if not isinstance(card["speech"], dict):
        raise ValueError(f"{path}: speech must be an object")

    return card


def render_character(card: dict) -> str:
    # JSON preserves an explicit separation between fields.
    return json.dumps(
        card, ensure_ascii=False, indent=2, sort_keys=True
    )


def build_npc_prompt(
    scenario: dict,
    scenario_path: str | Path,
) -> tuple[str, list[tuple[Path, dict]]]:
    base = scenario.get("system")

    if not isinstance(base, str) or not base.strip():
        raise ValueError("Scenario requires a nonempty system prompt")

    references = scenario.get("character_cards", [])

    if not isinstance(references, list):
        raise ValueError("character_cards must be a list")

    if not references:
        return base, []

    loaded = []
    seen_ids = set()

    for reference in references:
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Invalid character card reference")

        path = (
            Path(scenario_path).resolve().parent / reference
        ).resolve()

        card = load_character_card(path)

        if card["id"] in seen_ids:
            raise ValueError(
                f"Duplicate character id: {card['id']}"
            )

        seen_ids.add(card["id"])
        loaded.append((path, card))

    sections = [
        base,
        "CHARACTER DEFINITIONS:",
        "The following cards are separate fictional people.",
        "Keep each person's appearance, voice, memories, "
        "knowledge and relationships distinct.",
        "Private knowledge belongs only to its named owner. "
        "Other characters must not act on it until they learn "
        "it through events within the story.",
    ]

    for _, card in loaded:
        sections.extend([
            f"CHARACTER: {card['name']}",
            render_character(card),
        ])

    return "\n\n".join(sections), loaded
