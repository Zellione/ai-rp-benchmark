import pytest

from rpbench.refusal_preset_batch import (
    experiment_directory,
    experiment_identity,
    pending_cases,
    validate_archive,
)


MODEL = {
    "id": "test/Model",
    "variant": "standard",
    "quantization": "Q6_K",
}


def make_identity(preset_sha="a" * 64, compat="alternating-chat-v1"):
    return experiment_identity(
        MODEL, "c" * 64, "voyage-v6",
        preset_sha, 100001, compat,
    )


def test_experiment_fingerprint_changes_with_preset(tmp_path):
    a = experiment_directory(tmp_path, make_identity())
    b = experiment_directory(
        tmp_path, make_identity(preset_sha="b" * 64)
    )
    assert a != b


def test_experiment_fingerprint_changes_with_compat(tmp_path):
    a = experiment_directory(tmp_path, make_identity())
    b = experiment_directory(
        tmp_path, make_identity(compat=None)
    )
    assert a != b


def test_empty_archive(tmp_path):
    plan = [{"id": "case-a", "scenario_sha256": "x" * 64}]
    assert validate_archive(
        tmp_path / "new", make_identity(), plan
    ) == {}


def test_unmanifested_existing_archive_rejected(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "result.json").write_text("{}")

    with pytest.raises(ValueError, match="lacks experiment manifest"):
        validate_archive(directory, make_identity(), [])


def test_pending_excludes_completed():
    plan = [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
    ]
    archived = {
        "a": [{"status": "complete"}],
        "b": [{"status": "truncated"}],
    }

    assert [
        row["id"]
        for row in pending_cases(plan, archived, False)
    ] == ["c"]

    assert [
        row["id"]
        for row in pending_cases(plan, archived, True)
    ] == ["b", "c"]


def test_complete_result_prevents_retry():
    plan = [{"id": "case-a"}]
    archived = {
        "case-a": [
            {"status": "failed"},
            {"status": "complete"},
        ],
    }
    assert pending_cases(plan, archived, True) == []
