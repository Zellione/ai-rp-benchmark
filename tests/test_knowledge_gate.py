"""Player-perspective scene boundary validation."""

import pytest

from rpbench.knowledge_gate import (
    observable_text,
    split_paragraphs,
    validate_gate_result,
)


def test_scene_boundaries_filter_private_dialogue():
    text = (
        "Elena leaves the archive.\n\n"
        "Mara and Voss speak privately.\n\n"
        "Finn discusses the secret records."
    )

    parts = split_paragraphs(text)

    result = {
        "labels": [
            {"id": 0, "visibility": "observable"},
            {"id": 1, "visibility": "offscreen"},
            {"id": 2, "visibility": "offscreen"},
        ],
    }

    labels = validate_gate_result(result, len(parts))

    assert observable_text(parts, labels) == (
        "Elena leaves the archive."
    )


def test_fully_offscreen_scene():
    parts = split_paragraphs(
        "Mara speaks privately.\n\nVoss replies."
    )
    labels = ["offscreen", "offscreen"]

    assert observable_text(parts, labels) == ""


def test_mixed_scene_conservatively_excluded():
    parts = ["Elena leaves while Mara reveals a secret."]
    assert observable_text(parts, ["mixed"]) == ""


@pytest.mark.parametrize("result", [
    {"labels": [{"id": 0, "visibility": "invalid"}]},
    {"labels": [{"id": 1, "visibility": "observable"}]},
    {"labels": [{"id": True, "visibility": "observable"}]},
    {"labels": []},
    {"labels": [{"id": 0, "visibility": "observable"}],
     "extra": True},
])
def test_invalid_gate_result_rejected(result):
    with pytest.raises(ValueError):
        validate_gate_result(result, 1)


def test_mismatched_paragraph_count_rejected():
    with pytest.raises(ValueError):
        observable_text(["first"], [])


def test_split_paragraphs_preserves_narrative_order():
    assert split_paragraphs(
        "First.\n\nSecond.\n\nThird."
    ) == ["First.", "Second.", "Third."]


def test_observable_text_uses_real_paragraph_breaks():
    parts = split_paragraphs(
        "Visible first.\n\nHidden secret.\n\nVisible second."
    )

    result = observable_text(
        parts,
        ["observable", "offscreen", "observable"],
    )

    assert result == "Visible first.\n\nVisible second."
    assert "\\n" not in result
