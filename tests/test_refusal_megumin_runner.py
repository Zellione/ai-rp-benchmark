import inspect

from rpbench.refusal_probe import run_probe


def test_run_probe_exposes_megumin_configuration():
    signature = inspect.signature(run_probe)

    assert "megumin_config" in signature.parameters
    assert "megumin_engine_directory" in signature.parameters

    assert signature.parameters["megumin_config"].default is None
    assert (
        signature.parameters["megumin_engine_directory"].default
        is None
    )
