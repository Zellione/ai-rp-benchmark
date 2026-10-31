import hashlib
import json

from rpbench.preset_adapter import (
    SillyTavernPresetAdapter,
    render_with_adapter,
)
from rpbench.prompt_models import PromptMessage
from rpbench.simulation_prompts import render_npc_chat_messages


def make_preset():
    return {
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Play {{char}} with {{user}}.",
            },
            {
                "identifier": "charDescription",
                "marker": True,
                "role": "system",
            },
            {
                "identifier": "chatHistory",
                "marker": True,
                "role": "system",
            },
        ],
        "prompt_order": [{
            "character_id": 100001,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "charDescription", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    }


def test_snapshot_matches_file_adapter(tmp_path):
    source = tmp_path / "preset.json"
    source.write_text(
        json.dumps(make_preset()), encoding="utf-8"
    )

    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    adapter = SillyTavernPresetAdapter(
        source,
        preset_id="test",
        allow_disabled_orphans=True,
    )

    history = [
        PromptMessage("user", "Hello"),
        PromptMessage("assistant", "Greetings"),
        PromptMessage("user", "Where is the key?"),
    ]
    markers = {"charDescription": "Mara guards the archive."}

    expected = render_with_adapter(
        adapter,
        history,
        markers=markers,
        character_name="Mara",
        user_name="Alex",
        character_id=100001,
    ).messages

    restored = SillyTavernPresetAdapter.from_snapshot(
        adapter.preset.to_dict(),
        digest,
        preset_id="test",
    )

    result = render_with_adapter(
        restored,
        history,
        markers=markers,
        character_name="Mara",
        user_name="Alex",
        character_id=100001,
    ).messages

    assert result == expected

    # Also verify integration through the simulation renderer.
    simulation = render_npc_chat_messages(
        [
            {"role": "system", "content": markers["charDescription"]},
            *[m.to_dict() for m in history],
        ],
        restored.preset,
        markers,
        character_name="Mara",
        user_name="Alex",
        character_id=100001,
        source_sha256=digest,
    )

    assert simulation == expected

    # Snapshot does not require the original preset file.
    source.unlink()

    assert render_with_adapter(
        restored,
        history,
        markers=markers,
        character_name="Mara",
        user_name="Alex",
        character_id=100001,
    ).messages == expected
