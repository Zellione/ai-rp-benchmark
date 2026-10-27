"""Quest opportunity setting survives evaluator validation."""

import pytest

from rpbench.evaluator_runtime import validate_evaluator_config


BASE = {
    "mode": "llm",
    "id": "quest-config-test",
    "backend": {
        "name": "llamacpp",
        "base_url": "http://127.0.0.1:8081",
        "model": "mock",
    },
    "sampling": {"max_tokens": 1024},
}


def test_opportunity_opt_in_survives_validation():
    settings = {
        **BASE,
        "quest_opportunity_enabled": True,
    }

    validated = validate_evaluator_config(settings)

    assert validated["quest_opportunity_enabled"] is True
    assert validated is not settings


def test_opportunity_disabled_by_default():
    validated = validate_evaluator_config(BASE)

    assert validated.get(
        "quest_opportunity_enabled", False
    ) is False


@pytest.mark.parametrize("invalid", [
    1, 0, None, "true", [], {},
])
def test_opportunity_requires_boolean(invalid):
    with pytest.raises(ValueError):
        validate_evaluator_config({
            **BASE,
            "quest_opportunity_enabled": invalid,
        })
