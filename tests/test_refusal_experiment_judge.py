from pathlib import Path

from rpbench.refusal_experiment_judge import judge_commands


def test_both_judges_are_scheduled():
    experiment = {
        "directory": Path("/tmp/example"),
    }

    commands = judge_commands(
        experiment,
        Path("/tmp/judge.json"),
        Path("/tmp/output"),
        False,
    )

    assert [name for name, _ in commands] == [
        "refusal", "fulfillment"
    ]
    assert "--allow-incomplete" in commands[0][1]
    assert "--allow-incomplete" not in commands[1][1]


def test_retry_is_forwarded_to_both_judges():
    commands = judge_commands(
        {"directory": Path("/tmp/example")},
        Path("/tmp/judge.json"),
        Path("/tmp/output"),
        True,
    )

    assert all(
        "--retry-failed" in command
        for _, command in commands
    )
