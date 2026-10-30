"""Strict, data-only resolver for historical Megumin V9 slots."""

import re
from dataclasses import dataclass, field

SLOT = re.compile(r"\[\[[^\[\]\r\n]+\]\]")


class MeguminSlotError(ValueError):
    pass


@dataclass
class MeguminResolver:
    values: dict[str, str]
    provenance: dict[str, str] = field(default_factory=dict)

    def render(self, text: str, *, identifier: str = "<unknown>") -> str:
        if not isinstance(text, str):
            raise TypeError("Megumin content must be text")

        def expand(value: str, stack: tuple[str, ...]) -> str:
            def replace(match):
                slot = match.group()

                if slot in stack:
                    raise MeguminSlotError(
                        f"Cyclic slot reference: {' -> '.join((*stack, slot))}"
                    )
                if slot not in self.values:
                    raise MeguminSlotError(
                        f"{identifier}: undefined slot {slot}"
                    )

                replacement = self.values[slot]
                if not isinstance(replacement, str):
                    raise MeguminSlotError(
                        f"{identifier}: non-text slot {slot}"
                    )

                return expand(replacement, (*stack, slot))

            return SLOT.sub(replace, value)

        result = expand(text, ())
        if SLOT.search(result):
            raise MeguminSlotError(
                f"{identifier}: unresolved slots remain"
            )
        return result


def make_engine_slots(engine: dict, cot: dict) -> MeguminResolver:
    if engine.get("id") != "v9-core":
        raise ValueError("Expected historical v9-core engine")

    if cot.get("id") != "cot-v9-english":
        raise ValueError("Expected historical V9 English CoT")

    values = {}
    provenance = {}

    for index in range(1, 7):
        key = f"[[prompt{index}]]"
        values[key] = engine.get(f"p{index}") or ""
        provenance[key] = f"engine.v9-core.p{index}"

    cot_text = cot.get("content")
    if not isinstance(cot_text, str) or not cot_text.strip():
        raise ValueError("Empty historical CoT")

    # Historical buildBaseDict(), thinkingV2=False.
    values["[[THINK]]"] = f"<think>\n{cot_text}\n</think>"
    provenance["[[THINK]]"] = "cot-v9-english via historical THINK wrapper"

    # CoT is delivered through THINK, not inserted a second time.
    values["[[COT]]"] = ""
    provenance["[[COT]]"] = "historical duplicate suppression"

    return MeguminResolver(values, provenance)


def make_profile_resolver(
    engine: dict,
    cot: dict,
    config: dict,
) -> MeguminResolver:
    """Build the explicitly scoped historical V9 benchmark profile."""
    if config["engine"] != "v9-core":
        raise ValueError("Unsupported Megumin engine")
    if config["cot"] != "cot-v9-english":
        raise ValueError("Unsupported Megumin CoT")

    profile = config["profile"]

    if not profile["cotEnabled"]:
        raise ValueError("CoT-disabled variant requires a separate experiment")
    if profile.get("thinkingV2"):
        raise ValueError("thinkingV2 is not implemented")
    if profile.get("userPronouns") != "off":
        raise ValueError("Custom pronouns are not implemented")
    if profile.get("addons") or profile.get("blocks"):
        raise ValueError("Additional modules are not implemented")
    if any(profile.get(name, {}).get("enabled") for name in (
        "storyPlan", "imageGen", "dnRatio", "onomatopoeia"
    )):
        raise ValueError("Optional module enabled")

    if profile.get("toggles", {}).get("ooc"):
        raise ValueError("OOC toggle enabled")
    if profile.get("toggles", {}).get("control"):
        raise ValueError("Control toggle enabled")

    resolver = make_engine_slots(engine, cot)
    values = resolver.values
    provenance = resolver.provenance

    # These are wiped by the historical V8/V9 handling.
    historical_blank = (
        "main", "OOC", "control", "aiprompt"
    )

    # Explicitly disabled features for the RPBench profile.
    disabled = (
        "DNRATIO", "DN", "COLOR", "Direct", "MVU",
        "Short-memory", "long-Memory",
        "banlist", "combat", "cyoa", "death",
        "img1", "infoblock", "npc list",
        "npc_dossier", "npc_events",
        "npc_inner_chatter", "onomato",
        "storyplan", "storytracker",
        "pronouns",
    )

    for name in historical_blank:
        key = f"[[{name}]]"
        values[key] = ""
        provenance[key] = "historical V9 suppression"

    for name in disabled:
        key = f"[[{name}]]"
        values[key] = ""
        provenance[key] = "explicitly disabled benchmark feature"

    language = profile["userLanguage"].strip().upper()
    if not language:
        language = "ENGLISH"

    values["[[Language]]"] = (
        "[LANGUAGE RULE]\\n"
        f"ALL OUTPUT EXCEPT THINKING MUST BE IN {language} ONLY."
    )
    provenance["[[Language]]"] = "historical buildBaseDict language rule"

    limits = profile["v9Limits"]
    for key, field in (
        ("v9_lean_min", "leanMin"),
        ("v9_lean_max", "leanMax"),
        ("v9_full_min", "fullMin"),
        ("v9_full_max", "fullMax"),
    ):
        values[f"[[{key}]]"] = str(limits[field])
        provenance[f"[[{key}]]"] = f"profile.v9Limits.{field}"

    values["[[count]]"] = ""
    provenance["[[count]]"] = "V9 ignores legacy word count"

    return resolver
