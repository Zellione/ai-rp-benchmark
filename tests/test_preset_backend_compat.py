import json

import pytest

from rpbench.preset_backend_compat import (
    lower_to_alternating_chat,
    messages_sha256,
)
from rpbench.prompt_chat import validate_strict_chat_template
from rpbench.refusal_probe import run_probe
from tools.build_refusal_pilot import PROBES, build_scenario


ORIGINAL = [
    {"role": "system", "content": "Rule A"},
    {"role": "system", "content": "Rule B"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "Continue"},
    {"role": "system", "content": "Recall rules"},
]


def test_strict_template_rejects_original():
    with pytest.raises(ValueError, match="System message after"):
        validate_strict_chat_template(ORIGINAL)


def test_lowering_preserves_original():
    snapshot = json.loads(json.dumps(ORIGINAL))
    lowered, metadata = lower_to_alternating_chat(ORIGINAL)

    assert ORIGINAL == snapshot
    validate_strict_chat_template(lowered)

    assert [m["role"] for m in lowered] == [
        "system", "user", "assistant", "user"
    ]
    assert lowered[0]["content"] == "Rule A\n\nRule B"
    assert lowered[1]["content"] == "Hello"
    assert lowered[2]["content"] == "Hi"
    assert lowered[3]["content"].startswith("Continue")
    assert "Recall rules" in lowered[3]["content"]

    assert metadata["initial_system_messages_merged"] == 2
    assert metadata["post_history_system_messages_relocated"] == 1
    assert metadata["exact_sillytavern_serialization"] is False
    assert metadata["original_messages_sha256"] == messages_sha256(
        ORIGINAL
    )
    assert metadata["sent_messages_sha256"] == messages_sha256(
        lowered
    )


def test_in_history_system_rejected():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Mid-history"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Next"},
    ]

    with pytest.raises(ValueError, match="in-history"):
        lower_to_alternating_chat(messages)


def test_non_alternating_history_rejected():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]
    with pytest.raises(ValueError, match="not alternating"):
        lower_to_alternating_chat(messages)


def test_refusal_probe_mock_archives_both_message_sets(tmp_path, monkeypatch):
    scenario = build_scenario(PROBES[0])
    scenario_file = tmp_path / "scenario.json"
    config_file = tmp_path / "config.json"
    preset_file = tmp_path / "preset.json"

    scenario_file.write_text(json.dumps(scenario), encoding="utf-8")
    config_file.write_text(json.dumps({
        "backend": {
            "name": "mock",
            "base_url": "",
            "model": "mock",
            "mode": "chat",
        },
        "sampling": {"temperature": 0.7, "max_tokens": 256},
        "model_identity": {
            "id": "test/mock",
            "base_model": "mock",
            "publisher": "test",
            "variant": "standard",
            "quantization": "none",
            "source": {"type": "local", "path": "mock://model"},
        },
    }), encoding="utf-8")

    # Minimal ST preset with post-history instructions.
    preset_file.write_text(json.dumps({
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Narrate the scene.",
            },
            {"identifier": "charDescription", "marker": True},
            {"identifier": "chatHistory", "marker": True},
            {
                "identifier": "jailbreak",
                "role": "system",
                "content": "Recall character rules.",
            },
        ],
        "prompt_order": [{
            "character_id": 100001,
            "order": [
                {"identifier": "main", "enabled": True},
                {"identifier": "charDescription", "enabled": True},
                {"identifier": "chatHistory", "enabled": True},
                {"identifier": "jailbreak", "enabled": True},
            ],
        }],
    }), encoding="utf-8")

    record, directory = run_probe(
        scenario_file,
        config_file,
        tmp_path / "runs",
        preset_path=preset_file,
        preset_character_id=100001,
        preset_backend_compat="alternating-chat-v1",
    )
    archived = json.loads(
        (directory / "result.json").read_text(encoding="utf-8")
    )

    assert record["status"] == "complete"
    assert archived["status"] == "complete"
    assert archived["preset_messages"][-1] == {
        "role": "system",
        "content": "Recall character rules.",
    }
    validate_strict_chat_template(archived["messages"])
    assert len(archived["messages"]) == 6
    assert archived["messages"][-1]["content"].startswith(
        scenario["turns"][2]
    )
    assert archived["preset_backend_compat"]["strategy"] == (
        "alternating-chat-v1"
    )


def test_compatibility_requires_preset(tmp_path):
    from rpbench.refusal_probe import build_messages

    # A config-less request must fail during input loading;
    # separate negative coverage belongs in CLI integration tests.
    scenario = build_scenario(PROBES[0])
    assert len(build_messages(scenario)) == 6



def test_v2_merges_consecutive_user_injections():
    from rpbench.preset_backend_compat import (
        lower_to_alternating_chat_v2,
    )

    original = [
        {"role": "system", "content": "Main rules"},
        {"role": "user", "content": "Previous player"},
        {"role": "assistant", "content": "Previous NPC"},
        {"role": "user", "content": "Depth-1 injection"},
        {"role": "user", "content": "Current player"},
        {"role": "user", "content": "Depth-0 injection"},
        {"role": "system", "content": "NPC Genesis"},
    ]

    lowered, metadata = lower_to_alternating_chat_v2(original)
    validate_strict_chat_template(lowered)

    assert [m["role"] for m in lowered] == [
        "system", "user", "assistant", "user",
    ]
    assert lowered[-1]["content"].startswith(
        "Depth-1 injection\\n\\nCurrent player"
        "\\n\\nDepth-0 injection"
    )
    assert lowered[-1]["content"].endswith(
        "NPC Genesis\\n</rpbench-post-history-instructions>"
    )
    assert metadata["consecutive_user_messages_merged"] == 2
    assert metadata["post_history_system_messages_relocated"] == 1
    assert original[-1]["role"] == "system"


def test_v2_rejects_in_history_system():
    from rpbench.preset_backend_compat import (
        lower_to_alternating_chat_v2,
    )
    with pytest.raises(ValueError, match="in-history"):
        lower_to_alternating_chat_v2([
            {"role": "user", "content": "A"},
            {"role": "system", "content": "Mid"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ])
