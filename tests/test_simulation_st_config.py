"""Tests for reproducible opt-in NPC prompt configuration."""

import json

import pytest

from rpbench.simulation_st_config import (
    load_st_npc_config,
    render_st_npc_request,
)


def make_preset():
    return {
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Act as {{char}}.",
            },
            {
                "identifier": "rpbenchContext",
                "role": "system",
                "marker": True,
            },
            {
                "identifier": "chatHistory",
                "role": "system",
                "marker": True,
            },
        ],
        "prompt_order": [{
            "character_id": 100000,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "rpbenchContext", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    }


def settings():
    return {
        "st_chat_npc": {
            "preset": "preset.json",
            "character_name": "Elara",
            "user_name": "Rowan",
            "context_marker": "rpbenchContext",
        },
    }


def test_legacy_mode_unchanged(tmp_path):
    assert load_st_npc_config(
        {}, tmp_path / "config.json", "Legacy context"
    ) is None

    messages = [
        {"role": "system", "content": "Legacy context"},
        {"role": "user", "content": "Hello"},
    ]

    assert render_st_npc_request(messages, None) == messages


def test_preset_is_snapshotted(tmp_path):
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(make_preset()))

    archived = load_st_npc_config(
        settings(), tmp_path / "config.json",
        "Secret character knowledge",
    )

    # The original file can disappear: archived runs must
    # still render the same prompt.
    path.unlink()

    result = render_st_npc_request(
        [
            {
                "role": "system",
                "content": "Secret character knowledge",
            },
            {"role": "user", "content": "Hello"},
        ],
        archived,
    )

    assert result == [
        {"role": "system", "content": "Act as Elara."},
        {
            "role": "system",
            "content": "Secret character knowledge",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_missing_context_marker_rejected(tmp_path):
    preset = make_preset()
    preset["prompts"][1]["marker"] = False
    (tmp_path / "preset.json").write_text(json.dumps(preset))

    with pytest.raises(ValueError, match="Context marker"):
        load_st_npc_config(
            settings(), tmp_path / "config.json", "Context"
        )


def test_invalid_configuration_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_st_npc_config(
            {"st_chat_npc": {"preset": "x"}},
            tmp_path / "config.json",
            "Context",
        )


def test_invalid_archived_renderer_rejected():
    with pytest.raises(ValueError):
        render_st_npc_request(
            [{"role": "system", "content": "Context"}],
            {"mode": "unknown"},
        )
