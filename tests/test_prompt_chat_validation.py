import pytest

from rpbench.prompt_chat import validate_strict_chat_template


def test_valid_mistral_history():
    validate_strict_chat_template([
        {"role": "system", "content": "Character card"},
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": "Greeting"},
        {"role": "user", "content": "Reply"},
    ])


def test_system_after_history_rejected():
    with pytest.raises(ValueError, match="System message"):
        validate_strict_chat_template([
            {"role": "system", "content": "Character card"},
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "Greeting"},
            {"role": "system", "content": "Post-history"},
            {"role": "user", "content": "Reply"},
        ])


def test_assistant_first_rejected():
    with pytest.raises(ValueError, match="begin with a user"):
        validate_strict_chat_template([
            {"role": "system", "content": "Character card"},
            {"role": "assistant", "content": "Greeting"},
        ])


def test_consecutive_users_rejected():
    with pytest.raises(ValueError, match="Consecutive user"):
        validate_strict_chat_template([
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ])
