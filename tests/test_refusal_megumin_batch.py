from rpbench.refusal_megumin_batch import (
    canonical_digest,
    pending_cases,
    experiment_directory,
)


def test_fingerprint_stable_under_key_reordering():
    a = {"engine": "one", "cot": "two"}
    b = {"cot": "two", "engine": "one"}
    assert canonical_digest(a) == canonical_digest(b)


def test_engine_change_produces_new_deployment(tmp_path):
    identity = {
        "model_identity": {
            "id": "test",
            "variant": "base",
            "quantization": "Q8",
        },
        "engine_sha256": "a" * 64,
    }
    first = experiment_directory(tmp_path, identity)

    changed = {
        **identity,
        "engine_sha256": "b" * 64,
    }
    second = experiment_directory(tmp_path, changed)

    assert first != second


def test_completed_cases_not_repeated():
    plan = [{"id": "one"}, {"id": "two"}]
    archived = {
        "one": [{"status": "complete"}],
        "two": [{"status": "truncated"}],
    }
    assert pending_cases(plan, archived, False) == []
    assert pending_cases(plan, archived, True) == [plan[1]]
