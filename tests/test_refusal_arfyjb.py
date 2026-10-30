import hashlib
import json

import pytest

from rpbench.refusal_arfyjb import (
    ArfyJBConfigurationError,
    build_arfyjb_messages,
    load_arfyjb_source,
)


def fixture(tmp_path):
    parts = {
        name: {
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        for name, content in {
            "main": "Main rules",
            "post": "Post rules",
            "cyoa": "Choices",
            "final": "Assistant prefill",
        }.items()
    }
    path = tmp_path / "arfyjb.json"
    path.write_text(json.dumps({
        "id": "arfyjb-universal-source",
        "source_url": "https://rentry.org/ArfyJB",
        "source_html_sha256": "a" * 64,
        "components": parts,
        "benchmark_defaults": {
            "main": True,
            "post": True,
            "cyoa": False,
            "final": False,
            "assistant_prefill": False,
        },
    }))
    return path


def scenario():
    return {
        "id": "fixture",
        "category": "benign_trigger",
        "expected_behavior": "continue",
        "probe_turn": 3,
        "system": "Scenario rules",
        "turns": ["One", "Two", "Three"],
        "fixed_history": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Reply one"},
            {"role": "user", "content": "Two"},
            {"role": "assistant", "content": "Reply two"},
        ],
    }


def test_arfyjb_render(tmp_path):
    messages, metadata = build_arfyjb_messages(
        scenario(), source_path=fixture(tmp_path)
    )

    assert [m["role"] for m in messages] == [
        "system", "user", "assistant",
        "user", "assistant", "user"
    ]
    assert messages[1]["content"] == "One"
    assert messages[4]["content"] == "Reply two"
    assert messages[-1]["content"].endswith("Three")
    assert "Post rules" in messages[-1]["content"]
    assert "Scenario rules" in messages[0]["content"]
    assert metadata["prefill_applied"] is False


def test_rejects_changed_source_component(tmp_path):
    path = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["components"]["main"]["content"] = "Modified"
    path.write_text(json.dumps(data))

    with pytest.raises(
        ArfyJBConfigurationError, match="fingerprint"
    ):
        load_arfyjb_source(path)


def test_rejects_unsupported_modules(tmp_path):
    path = fixture(tmp_path)
    data = json.loads(path.read_text())
    data["benchmark_defaults"]["cyoa"] = True
    path.write_text(json.dumps(data))

    with pytest.raises(
        ArfyJBConfigurationError, match="module selection"
    ):
        build_arfyjb_messages(scenario(), source_path=path)
