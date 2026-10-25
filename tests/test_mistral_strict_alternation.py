from rpbench.simulation import make_histories


def test_mistral_npc_opening_alternation():
    player, npc = make_histories(
        "Player instructions",
        "NPC instructions",
        "Opening situation",
        "NPC opening",
        strict_alternation=True,
    )

    assert [m["role"] for m in npc] == [
        "system",
        "user",
        "assistant",
    ]

    assert npc[1]["content"] == "Opening situation"
    assert npc[2]["content"] == "NPC opening"

    assert [m["role"] for m in player] == [
        "system",
        "user",
    ]

    assert player[1]["content"] == "NPC opening"


def test_legacy_npc_opening_unchanged():
    _, npc = make_histories(
        "Player instructions",
        "NPC instructions",
        "Opening situation",
        "NPC opening",
    )

    assert [m["role"] for m in npc] == [
        "system",
        "assistant",
    ]


def test_without_npc_opening():
    _, npc = make_histories(
        "Player instructions",
        "NPC instructions",
        "Opening situation",
        strict_alternation=True,
    )

    assert [m["role"] for m in npc] == ["system"]
