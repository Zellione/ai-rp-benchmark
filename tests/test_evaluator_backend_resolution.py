"""Evaluator defaults to RP endpoint unless explicitly overridden."""

from copy import deepcopy

import pytest

from rpbench.evaluator_runtime import resolve_evaluator_config


MAIN = {
    "name": "llamacpp",
    "base_url": "http://127.0.0.1:8080",
    "model": "qwen-local",
    "mode": "chat",
    "timeout": 300,
}

EVALUATOR = {
    "mode": "llm",
    "id": "quest-evaluator",
    "sampling": {"max_tokens": 1024},
}


def test_inherits_main_endpoint():
    original = deepcopy(EVALUATOR)

    resolved = resolve_evaluator_config(EVALUATOR, MAIN)

    assert resolved["backend"] == {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:8080",
        "model": "qwen-local",
        "timeout": 300,
    }
    assert EVALUATOR == original


def test_explicit_endpoint_wins():
    secondary = {
        "name": "vllm",
        "base_url": "http://127.0.0.1:8082",
        "model": "quest-model",
    }

    resolved = resolve_evaluator_config({
        **EVALUATOR,
        "backend": secondary,
    }, MAIN)

    assert resolved["backend"] == secondary
    assert resolved["backend"]["model"] != MAIN["model"]


def test_main_auth_environment_is_preserved():
    resolved = resolve_evaluator_config(
        EVALUATOR,
        {
            **MAIN,
            "api_key_env": "RPBENCH_MAIN_API_KEY",
        },
    )

    assert resolved["backend"]["api_key_env"] == (
        "RPBENCH_MAIN_API_KEY"
    )


def test_incompatible_main_requires_explicit_evaluator():
    with pytest.raises(ValueError, match="chat endpoint"):
        resolve_evaluator_config(
            EVALUATOR,
            {**MAIN, "mode": "completion"},
        )


def test_missing_evaluator_remains_disabled():
    assert resolve_evaluator_config(None, MAIN) is None


def test_explicit_backend_needs_no_main_backend():
    secondary = {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:8081",
        "model": "deckard",
    }

    resolved = resolve_evaluator_config(
        {**EVALUATOR, "backend": secondary},
        None,
    )

    assert resolved["backend"] == secondary
