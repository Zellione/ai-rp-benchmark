from rpbench.refusal_experiment_run import (
    experiment_dir,
    fingerprint,
    probe_options,
    select_pending,
)


def test_identity_fingerprint_deterministic():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint(
        {"b": 2, "a": 1}
    )


def test_changed_output_budget_changes_directory(tmp_path):
    identity = {
        "model_key": "test",
        "preset_key": "baseline",
        "output_tokens": 4096,
    }
    first = experiment_dir(tmp_path, identity)
    second = experiment_dir(
        tmp_path, {**identity, "output_tokens": 8192}
    )
    assert first != second


def test_archived_complete_not_repeated():
    plan = [{"id": "one"}, {"id": "two"}]
    archived = {
        "one": [{"status": "complete"}],
        "two": [{"status": "truncated"}],
    }

    assert select_pending(plan, archived) == []
    assert select_pending(plan, archived, True) == [plan[1]]


def test_adapter_routing():
    assert probe_options({"adapter": "baseline"}) == {}

    result = probe_options({
        "adapter": "arfyjb",
        "source": "arfyjb.json",
    })
    assert result == {"arfyjb_source": "arfyjb.json"}

    result = probe_options({
        "adapter": "megumin-v9",
        "config": "profile.json",
        "engine_directory": "engine",
    })
    assert result["megumin_config"] == "profile.json"
    assert result["megumin_engine_directory"] == "engine"
