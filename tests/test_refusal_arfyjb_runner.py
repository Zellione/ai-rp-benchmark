import inspect

from rpbench.refusal_probe import run_probe


def test_run_probe_supports_arfyjb():
    signature = inspect.signature(run_probe)

    assert "arfyjb_source" in signature.parameters
    assert signature.parameters["arfyjb_source"].default is None
