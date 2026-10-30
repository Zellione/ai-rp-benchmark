import hashlib
import json

import pytest

from rpbench.refusal_experiment_report import (
    md_code,
    verdicts_for,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def test_markdown_preserves_complete_text():
    content = "Erste Zeile\n\nZweite Zeile\n"
    rendered = md_code(content)

    assert "Erste Zeile\n\nZweite Zeile" in rendered


def test_markdown_handles_nested_backticks():
    content = "Ein Codeblock:\n```python\nprint(1)\n```"
    rendered = md_code(content)

    assert rendered.startswith("````text\n")
    assert content in rendered


def test_judge_selection_after_invalid_attempt(tmp_path):
    probe = tmp_path / "result.json"
    probe.write_text('{"status":"complete"}', encoding="utf-8")
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()

    directory = tmp_path / "judgments"
    write_json(directory / "001.json", {
        "valid": False,
        "probe_sha256": digest,
    })
    write_json(directory / "002.json", {
        "valid": True,
        "probe_sha256": digest,
        "verdict": {"model_refusal": False},
    })

    selected = verdicts_for(probe, directory, "refusal")

    assert selected["verdict"]["model_refusal"] is False
    assert selected["invalid_attempts"] == 1


def test_judge_rejects_wrong_probe_fingerprint(tmp_path):
    probe = tmp_path / "result.json"
    probe.write_text('{"status":"complete"}', encoding="utf-8")

    directory = tmp_path / "judgments"
    write_json(directory / "001.json", {
        "valid": True,
        "probe_sha256": "0" * 64,
        "verdict": {"model_refusal": False},
    })

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verdicts_for(probe, directory, "refusal")


def test_judge_rejects_duplicate_valid_verdicts(tmp_path):
    probe = tmp_path / "result.json"
    probe.write_text('{"status":"complete"}', encoding="utf-8")
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()

    directory = tmp_path / "judgments"
    for name in ("001.json", "002.json"):
        write_json(directory / name, {
            "valid": True,
            "probe_sha256": digest,
            "verdict": {"model_refusal": False},
        })

    with pytest.raises(ValueError, match="exactly one valid"):
        verdicts_for(probe, directory, "refusal")
