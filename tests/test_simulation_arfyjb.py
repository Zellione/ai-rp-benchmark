import copy

import pytest

from rpbench.prompt_chat import validate_strict_chat_template
from rpbench.simulation_arfyjb import render_arfyjb_npc
from rpbench.simulation_st_config import render_st_npc_request


def snapshot():
    return {
        "mode": "arfyjb_universal",
        "source": {
            "components": {
                "main": {"content": "Main ArfyJB instructions"},
                "post": {"content": "Post ArfyJB instructions"},
            }
        },
    }


def history():
    return [
        {"role": "system", "content": "You are Mara"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Greetings"},
        {"role": "user", "content": "Where is the key?"},
    ]


def test_arfyjb_preserves_history_and_last_player_turn():
    messages = history()
    original = copy.deepcopy(messages)

    result = render_arfyjb_npc(messages, snapshot())

    validate_strict_chat_template(result)
    assert messages == original
    assert result[1:-1] == messages[1:-1]
    assert result[-1]["content"].endswith("Where is the key?")
    assert "Main ArfyJB instructions" in result[0]["content"]
    assert "Post ArfyJB instructions" in result[-1]["content"]


def test_common_simulation_renderer_dispatches_arfyjb():
    result = render_st_npc_request(history(), snapshot())
    assert "Post ArfyJB instructions" in result[-1]["content"]


def test_arfyjb_rejects_incomplete_history():
    with pytest.raises(ValueError):
        render_arfyjb_npc(
            [{"role": "system", "content": "Mara"}],
            snapshot(),
        )


def test_arfyjb_rejects_invalid_snapshot():
    with pytest.raises(ValueError):
        render_arfyjb_npc(
            history(), {"mode": "unknown"}
        )
