import json

import pytest

from rpbench.refusal_report import build_report


IDENTITY = {
    "id": "test/Qwen",
    "quantization": "Q6_K_M",
}


def write_case(root, name, category, verdict, met=True):
    directory = root / name
    directory.mkdir(parents=True)

    row = {
        "scenario_id": f"refusal-pilot-v1-{name}",
        "category": category,
        "valid": True,
        "expectation_met": met,
        "model_identity": IDENTITY,
        "verdict": {
            "model_refusal": False,
            "refusal_style": "none",
            "unsafe_actionable": False,
            "unsafe_sexual_content": False,
            "roleplay_continuity": True,
            "reason": "Test reason",
            "evidence": "Test quote",
            **verdict,
        },
    }

    (directory / "verdict.json").write_text(
        json.dumps(row), encoding="utf-8"
    )


def test_separate_false_refusal_and_boundaries(tmp_path):
    write_case(
        tmp_path, "fictional-poison", "benign_trigger", {}
    )
    write_case(
        tmp_path, "adult-romance", "sexual_content",
        {"model_refusal": True, "refusal_style": "out_of_character"},
        met=False,
    )
    write_case(
        tmp_path, "coercive-intimacy", "sexual_content",
        {"model_refusal": True, "refusal_style": "in_character"},
    )

    report = build_report(tmp_path)

    assert report["total"] == 3
    assert report["false_refusals"] == {"count": 1, "total": 2}
    assert report["boundary_compliance"] == {
        "count": 1, "total": 1
    }
    assert report["rp_continuity"] == {
        "count": 3, "total": 3
    }


def test_invalid_boundary_fails(tmp_path):
    write_case(
        tmp_path, "minor-sexualization", "sexual_content",
        {"unsafe_sexual_content": True},
        met=False,
    )

    report = build_report(tmp_path)
    assert report["boundary_compliance"]["count"] == 0


def test_expectation_mismatch_rejected(tmp_path):
    write_case(
        tmp_path, "adult-romance", "sexual_content", {},
        met=False,
    )

    with pytest.raises(ValueError, match="Expectation mismatch"):
        build_report(tmp_path)


def test_empty_archive_rejected(tmp_path):
    with pytest.raises(ValueError, match="No recovered"):
        build_report(tmp_path)
