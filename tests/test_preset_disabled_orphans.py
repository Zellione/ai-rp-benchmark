import json

import pytest

from rpbench.prompt_presets import SillyTavernChatPreset
from rpbench.preset_adapter import SillyTavernPresetAdapter


def make_payload(orphan_enabled=False):
    return {
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Hello",
            },
            {
                "identifier": "chatHistory",
                "marker": True,
            },
        ],
        "prompt_order": [{
            "character_id": 100001,
            "order": [
                {"identifier": "main", "enabled": True},
                {
                    "identifier": "deleted-prompt",
                    "enabled": orphan_enabled,
                },
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    }


def test_strict_rejects_disabled_orphan():
    with pytest.raises(ValueError, match="Unknown ordered"):
        SillyTavernChatPreset(make_payload())


def test_compatibility_skips_disabled_orphan():
    preset = SillyTavernChatPreset(
        make_payload(),
        allow_disabled_orphans=True,
    )

    ordered = preset.ordered_prompts(100001)

    assert [item["identifier"] for item in ordered] == [
        "main",
        "chatHistory",
    ]

    # Lossless export retains the original reference.
    original = preset.to_dict()
    assert len(original["prompt_order"][0]["order"]) == 3


def test_active_orphan_still_rejected():
    with pytest.raises(ValueError, match="Unknown ordered"):
        SillyTavernChatPreset(
            make_payload(orphan_enabled=True),
            allow_disabled_orphans=True,
        )


def test_adapter_imports_compatible_preset(tmp_path):
    file = tmp_path / "preset.json"
    file.write_text(json.dumps(make_payload()))

    adapter = SillyTavernPresetAdapter(
        file,
        preset_id="test",
        allow_disabled_orphans=True,
    )

    assert len(adapter.preset.ordered_prompts(100001)) == 2
