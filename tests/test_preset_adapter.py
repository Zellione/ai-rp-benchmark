import json

import pytest

from rpbench.preset_adapter import (
    IdentityPresetAdapter,
    SillyTavernPresetAdapter,
    render_with_adapter,
)
from rpbench.prompt_models import PromptMessage


HISTORY = [
    PromptMessage("system", "Original system"),
    PromptMessage("user", "Hello"),
]


def make_preset(tmp_path):
    path = tmp_path / "preset.json"
    path.write_text(json.dumps({
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "You are {{char}}.",
            },
            {
                "identifier": "chatHistory",
                "marker": True,
            },
        ],
        "prompt_order": [{
            "character_id": 100000,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
            ],
        }],
    }), encoding="utf-8")
    return path


def render(adapter):
    return render_with_adapter(
        adapter,
        HISTORY,
        markers={},
        character_name="Mara",
        user_name="Jonas",
    )


def test_identity_preserves_history():
    result = render(IdentityPresetAdapter())

    assert result.messages == [
        message.to_dict() for message in HISTORY
    ]
    assert result.applied_hooks == []


def test_sillytavern_order_and_macro(tmp_path):
    result = render(
        SillyTavernPresetAdapter(
            make_preset(tmp_path),
            preset_id="test-preset",
        )
    )

    assert result.messages[0] == {
        "role": "system",
        "content": "You are Mara.",
    }
    assert result.messages[-1] == {
        "role": "user",
        "content": "Hello",
    }
    assert result.applied_hooks == [
        "sillytavern_chat_render",
        "st_template_subset",
    ]
    assert len(result.source_sha256) == 64


def test_missing_marker_rejected(tmp_path):
    path = make_preset(tmp_path)
    data = json.loads(path.read_text())

    data["prompts"].append({
        "identifier": "worldInfoBefore",
        "marker": True,
    })
    data["prompt_order"][0]["order"].insert(1, {
        "identifier": "worldInfoBefore",
        "enabled": True,
    })
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing active"):
        render(
            SillyTavernPresetAdapter(
                path,
                preset_id="test",
            )
        )


def test_unsupported_position_rejected(tmp_path):
    path = make_preset(tmp_path)
    data = json.loads(path.read_text())
    data["prompts"][0]["injection_position"] = 7
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Unsupported preset"):
        render(
            SillyTavernPresetAdapter(
                path,
                preset_id="test",
            )
        )


def test_disabled_prompt_does_not_trigger_error(tmp_path):
    path = make_preset(tmp_path)
    data = json.loads(path.read_text())
    data["prompts"][0]["injection_position"] = 7
    data["prompt_order"][0]["order"][0]["enabled"] = False
    path.write_text(json.dumps(data))

    result = render(
        SillyTavernPresetAdapter(
            path,
            preset_id="test",
        )
    )
    assert result.messages == [
        message.to_dict() for message in HISTORY
    ]



def test_preset_squash_setting_loaded(tmp_path):
    path = make_preset(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["squash_system_messages"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = SillyTavernPresetAdapter(
        path, preset_id="squash-test"
    )
    result = render(adapter)

    assert adapter.squash_system_messages is True
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == (
        "You are Mara.\nOriginal system"
    )
    assert result.messages[1]["role"] == "user"


def test_explicit_squash_override(tmp_path):
    path = make_preset(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["squash_system_messages"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = SillyTavernPresetAdapter(
        path,
        preset_id="override-test",
        squash_system_messages=False,
    )

    assert adapter.squash_system_messages is False
    messages = render(adapter).messages

    assert len(messages) == 3
    assert [m["role"] for m in messages] == [
        "system", "system", "user"
    ]
    assert messages[0]["content"] == "You are Mara."
    assert messages[1]["content"] == "Original system"
    assert messages[2]["content"] == "Hello"
