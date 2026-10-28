import pytest

from rpbench.deployment_preflight import verify


def make_config(backend="llamacpp"):
    return {
        "backend": {
            "name": backend,
            "model": "test-model",
        },
        "model_identity": {
            "id": "example/Test-Model",
            "quantization": "Q6_K",
            "source": {
                "type": "local",
                "path": "/models/test.gguf",
            },
        },
    }


MODELS = {
    "data": [{"id": "test-model"}],
}


def test_llamacpp_model_verified():
    result = verify(
        make_config(),
        MODELS,
        {"model_path": "/models/test.gguf"},
    )
    assert result["verified_model_path"] is True


def test_wrong_api_model_rejected():
    with pytest.raises(ValueError, match="API model mismatch"):
        verify(
            make_config(),
            {"data": [{"id": "other-model"}]},
            {"model_path": "/models/test.gguf"},
        )


def test_wrong_model_path_rejected():
    with pytest.raises(ValueError, match="Loaded model mismatch"):
        verify(
            make_config(),
            MODELS,
            {"model_path": "/models/other.gguf"},
        )


def test_missing_llamacpp_properties_rejected():
    with pytest.raises(ValueError, match="Missing llama.cpp"):
        verify(make_config(), MODELS)


def test_ninfer_api_identity_verified():
    result = verify(
        make_config("ninfer"),
        MODELS,
    )
    assert result["verified_model_path"] is False
    assert result["backend"] == "ninfer"
