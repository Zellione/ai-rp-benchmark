import pytest

from rpbench.simulation import make_histories


def test_sillytavern_npc_first_history():
    player, npc = make_histories(
        "Player instructions",
        "NPC instructions",
        "Original scenario opening",
        "Elara's greeting",
        user_filler="[Start a new chat]",
    )

    assert [m["role"] for m in npc] == [
        "system",
        "user",
        "assistant",
    ]

    assert npc[1]["content"] == "[Start a new chat]"
    assert npc[2]["content"] == "Elara's greeting"

    assert [m["role"] for m in player] == [
        "system",
        "user",
    ]

    assert player[1]["content"] == "Elara's greeting"


def test_previous_mistral_format_preserved():
    _, npc = make_histories(
        "Player",
        "NPC",
        "Scenario opening",
        "NPC greeting",
        strict_alternation=True,
    )

    assert npc[1] == {
        "role": "user",
        "content": "Scenario opening",
    }


def test_legacy_format_preserved():
    _, npc = make_histories(
        "Player",
        "NPC",
        "Scenario opening",
        "NPC greeting",
    )

    assert [m["role"] for m in npc] == [
        "system",
        "assistant",
    ]


def test_filler_requires_nonempty_text():
    with pytest.raises(ValueError):
        make_histories(
            "Player",
            "NPC",
            "Scenario opening",
            "NPC greeting",
            user_filler=" ",
        )
