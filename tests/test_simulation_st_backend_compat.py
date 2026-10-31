import copy
import json

import pytest

from rpbench.simulation_st_config import (
    load_st_npc_config,
    render_st_npc_request,
)
from rpbench.prompt_chat import validate_strict_chat_template


def make_preset():
    return {
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "You are {{char}}.",
            },
            {
                "identifier": "charDescription",
                "role": "system",
                "marker": True,
            },
            {
                "identifier": "chatHistory",
                "role": "system",
                "marker": True,
            },
            {
                "identifier": "post",
                "role": "system",
                "content": "Remain in character.",
            },
        ],
        "prompt_order": [{
            "character_id": 100001,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "charDescription", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
                {"identifier": "post", "enabled": True},
                {"identifier": "missing-disabled", "enabled": False},
            ],
        }],
    }


def settings(strategy="alternating-chat-v1"):
    return {
        "st_chat_npc": {
            "preset": "preset.json",
            "character_name": "Mara",
            "user_name": "Alex",
            "context_marker": "charDescription",
            "character_id": 100001,
            "backend_compat": strategy,
        },
    }


def make_config(tmp_path, strategy="alternating-chat-v1"):
    (tmp_path / "preset.json").write_text(
        json.dumps(make_preset()), encoding="utf-8"
    )
    return load_st_npc_config(
        settings(strategy),
        tmp_path / "config.json",
        "Mara is suspicious.",
    )


def test_voyage_compatibility_produces_strict_chat(tmp_path):
    archived = make_config(tmp_path)
    messages = [
        {"role": "system", "content": "Mara is suspicious."},
        {"role": "user", "content": "Where is the key?"},
    ]
    original = copy.deepcopy(messages)

    result = render_st_npc_request(messages, archived)

    validate_strict_chat_template(result)
    assert messages == original
    assert result[0]["role"] == "system"
    assert result[-1]["role"] == "user"
    assert "Where is the key?" in result[-1]["content"]
    assert "Remain in character." in result[-1]["content"]


def test_compatibility_is_snapshotted(tmp_path):
    archived = make_config(tmp_path)
    assert archived["backend_compat"] == "alternating-chat-v1"

    (tmp_path / "preset.json").unlink()

    result = render_st_npc_request(
        [
            {"role": "system", "content": "Mara is suspicious."},
            {"role": "user", "content": "Hello"},
        ],
        archived,
    )
    validate_strict_chat_template(result)


def test_unsupported_strategy_rejected(tmp_path):
    (tmp_path / "preset.json").write_text(
        json.dumps(make_preset()), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="backend_compat"):
        load_st_npc_config(
            settings("unknown"),
            tmp_path / "config.json",
            "Context",
        )


def test_baseline_request_unchanged():
    messages = [
        {"role": "system", "content": "Mara"},
        {"role": "user", "content": "Hello"},
    ]
    assert render_st_npc_request(messages, None) == messages
