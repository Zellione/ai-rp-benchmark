from copy import deepcopy
from pathlib import Path

import pytest

from rpbench.prompt_presets import SillyTavernChatPreset


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "sillytavern"
    / "default-openai.json"
)


def test_official_preset_import_roundtrip():
    preset = SillyTavernChatPreset.from_file(FIXTURE)

    clone = SillyTavernChatPreset(preset.to_dict())

    assert clone.to_dict() == preset.to_dict()


def test_official_preset_has_history_marker():
    preset = SillyTavernChatPreset.from_file(FIXTURE)

    ordered = preset.ordered_prompts()
    identifiers = [item["identifier"] for item in ordered]

    assert "chatHistory" in identifiers
    assert "main" in identifiers
    assert "jailbreak" in identifiers

    history = next(
        item for item in ordered
        if item["identifier"] == "chatHistory"
    )

    assert history["marker"] is True


def test_official_preset_preserves_unknown_fields():
    preset = SillyTavernChatPreset.from_file(FIXTURE)
    original = preset.to_dict()

    original["rpbench_unknown_test"] = {
        "nested": [1, False, {"value": "preserved"}]
    }

    imported = SillyTavernChatPreset(original)

    assert imported.to_dict() == original


def test_payload_is_defensively_copied():
    original = {
        "prompts": [
            {"identifier": "main", "content": "Original"}
        ],
        "prompt_order": [
            {
                "character_id": 100000,
                "order": [
                    {"identifier": "main", "enabled": True}
                ],
            }
        ],
    }

    expected = deepcopy(original)
    preset = SillyTavernChatPreset(original)

    original["prompts"][0]["content"] = "Changed"
    assert preset.to_dict() == expected

    exported = preset.to_dict()
    exported["prompts"][0]["content"] = "Changed again"

    assert preset.to_dict() == expected


def test_disabled_prompt_preserved():
    preset = SillyTavernChatPreset({
        "prompts": [
            {"identifier": "main", "content": "A"},
            {"identifier": "optional", "content": "B"},
        ],
        "prompt_order": [
            {
                "character_id": 100000,
                "order": [
                    {"identifier": "main", "enabled": True},
                    {"identifier": "optional", "enabled": False},
                ],
            },
        ],
    })

    ordered = preset.ordered_prompts()

    assert [p["enabled"] for p in ordered] == [True, False]
    assert ordered[1]["content"] == "B"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompts": [], "prompt_order": "invalid"},
        {
            "prompts": [
                {"identifier": "main"},
                {"identifier": "main"},
            ],
            "prompt_order": [],
        },
        {
            "prompts": [{"identifier": "main"}],
            "prompt_order": [{
                "character_id": 100000,
                "order": [
                    {"identifier": "unknown", "enabled": True}
                ],
            }],
        },
        {
            "prompts": [{"identifier": "main"}],
            "prompt_order": [{
                "character_id": 100000,
                "order": [
                    {"identifier": "main", "enabled": "yes"}
                ],
            }],
        },
    ],
)
def test_invalid_presets_rejected(payload):
    with pytest.raises(ValueError):
        SillyTavernChatPreset(payload)
