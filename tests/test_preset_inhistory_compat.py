from copy import deepcopy

import pytest

from rpbench.preset_inhistory_compat import relocate_prefinal_system


def test_relocates_system_before_final_user():
    original = [
        {"role": "system", "content": "Engine"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Final"},
    ]
    snapshot = deepcopy(original)
    result, count = relocate_prefinal_system(original)

    assert count == 1
    assert [m["role"] for m in result] == [
        "system", "user", "assistant", "user"
    ]
    assert "Rules" in result[-1]["content"]
    assert result[-1]["content"].endswith("Final")
    assert original == snapshot


def test_without_injection_is_noop():
    messages = [
        {"role": "system", "content": "Engine"},
        {"role": "user", "content": "Final"},
    ]
    result, count = relocate_prefinal_system(messages)
    assert result == messages
    assert count == 0


def test_rejects_missing_final_user():
    with pytest.raises(ValueError, match="final user"):
        relocate_prefinal_system([
            {"role": "system", "content": "Engine"},
            {"role": "assistant", "content": "Reply"},
            {"role": "system", "content": "Rules"},
        ])


def test_multiple_initial_system_messages_preserved():
    messages = [
        {"role": "system", "content": "Engine"},
        {"role": "system", "content": "Additional rules"},
        {"role": "user", "content": "Final"},
    ]

    result, count = relocate_prefinal_system(messages)

    assert count == 0
    assert result == messages


def test_mirage_prefinal_injection_with_history():
    messages = [
        {"role": "system", "content": "Mirage engine"},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "system", "content": "Output rules"},
        {"role": "user", "content": "Final probe"},
    ]

    result, count = relocate_prefinal_system(messages)

    assert count == 1
    assert [m["role"] for m in result] == [
        "system", "user", "assistant", "user"
    ]
    assert "Output rules" in result[-1]["content"]
    assert result[-1]["content"].endswith("Final probe")
