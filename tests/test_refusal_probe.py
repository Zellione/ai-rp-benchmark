import json

import pytest

from rpbench.refusal_probe import (
    build_messages,
    run_probe,
    validate_scenario,
)
from tools.build_refusal_pilot import PROBES, build_scenario


@pytest.mark.parametrize("entry", PROBES, ids=lambda x: x[0])
def test_fixed_history(entry):
    scenario = build_scenario(entry)
    messages = build_messages(scenario)

    assert len(messages) == 6
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant",
        "user", "assistant", "user",
    ]
    assert messages[-1]["content"] == scenario["turns"][2]


def test_history_mismatch_rejected():
    scenario = build_scenario(PROBES[0])
    scenario["fixed_history"][0]["content"] = "Different text"

    with pytest.raises(ValueError, match="differs"):
        validate_scenario(scenario)


def test_missing_npc_history_rejected():
    scenario = build_scenario(PROBES[0])
    scenario["fixed_history"].pop()

    with pytest.raises(ValueError, match="structure"):
        validate_scenario(scenario)


def test_mock_runner_archives_exact_messages(tmp_path):
    scenario = build_scenario(PROBES[0])
    scenario_file = tmp_path / "scenario.json"
    config_file = tmp_path / "config.json"

    scenario_file.write_text(
        json.dumps(scenario), encoding="utf-8"
    )
    config_file.write_text(
        json.dumps({
            "backend": {
                "name": "mock",
                "base_url": "",
                "model": "mock",
                "mode": "chat",
            },
            "sampling": {
                "temperature": 0.7,
                "max_tokens": 256,
            },
            "model_identity": {
                "id": "test/mock",
                "base_model": "mock",
                "publisher": "test",
                "variant": "standard",
                "quantization": "none",
                "source": {
                    "type": "local",
                    "path": "mock://model",
                },
            },
        }),
        encoding="utf-8",
    )

    record, directory = run_probe(
        scenario_file, config_file, tmp_path / "runs"
    )

    archived = json.loads(
        (directory / "result.json").read_text()
    )

    assert record["status"] == "complete"
    assert archived["status"] == "complete"
    assert archived["model_identity"]["id"] == "test/mock"
    assert archived["messages"] == build_messages(scenario)
    assert archived["response"] == "[mock answer for turn 3]"
    assert len(archived["scenario_sha256"]) == 64
