"""Deterministic tests for repetition diagnostics."""

from rpbench.thinking_loop_benchmark import repetition_metrics


def test_empty_reasoning():
    result = repetition_metrics("")
    assert result["words"] == 0
    assert result["repeated_window_fraction"] == 0.0


def test_short_reasoning():
    result = repetition_metrics("Check the evidence.")
    assert result["max_window_occurrences"] == 0


def test_nonrepeating_reasoning():
    reasoning = " ".join(f"word{i}" for i in range(600))
    result = repetition_metrics(reasoning)
    assert result["repeated_window_fraction"] == 0.0
    assert result["max_window_occurrences"] == 1


def test_repetitive_reasoning():
    paragraph = " ".join(
        f"fact{i}" for i in range(48)
    )
    reasoning = " ".join([paragraph] * 40)

    result = repetition_metrics(reasoning)

    assert result["words"] > 300
    assert result["repeated_window_fraction"] > 0.8
    assert result["max_window_occurrences"] >= 4


def test_few_repetitions_not_dominant():
    reasoning = (
        " ".join(f"unique{i}" for i in range(500))
        + " "
        + " ".join(["repeat"] * 20)
    )

    result = repetition_metrics(reasoning)
    assert result["repeated_window_fraction"] < 0.35
