from rpbench.player_knowledge import (
    format_player_knowledge,
    initial_player_knowledge,
    replay_player_knowledge,
    update_player_knowledge,
)


RULES = [{
    "id": "archive-director",
    "npc_contains": "Director Vale",
    "fact": "An NPC mentioned Director Vale.",
}]


def test_initial_knowledge():
    knowledge = initial_player_knowledge({
        "known_facts": ["The archive is in Blackwood."],
    })

    assert knowledge["facts"][0]["status"] == "known"
    assert knowledge["facts"][0]["turn"] == 0


def test_npc_claim_is_recorded_as_reported():
    original = initial_player_knowledge({})
    updated = update_player_knowledge(
        original,
        2,
        "You should speak with Director Vale.",
        RULES,
    )

    assert original["facts"] == []
    assert updated["facts"][0]["status"] == "reported"
    assert updated["facts"][0]["turn"] == 2
    assert updated["facts"][0]["evidence"] == "Director Vale"


def test_duplicate_claim_is_not_added_twice():
    knowledge = initial_player_knowledge({})

    for turn in (1, 2, 3):
        knowledge = update_player_knowledge(
            knowledge, turn, "Director Vale", RULES
        )

    assert len(knowledge["facts"]) == 1
    assert knowledge["facts"][0]["turn"] == 1


def test_unmatched_text_does_not_add_knowledge():
    result = update_player_knowledge(
        initial_player_knowledge({}),
        1,
        "The archive is closed.",
        RULES,
    )

    assert result["facts"] == []


def test_replay_uses_npc_observations_only():
    observations = [
        {
            "turn": 1, "role": "player",
            "text": "Director Vale is here!",
        },
        {
            "turn": 1, "role": "npc",
            "text": "I cannot confirm that.",
        },
        {
            "turn": 2, "role": "npc",
            "text": "Ask Director Vale.",
        },
    ]

    result = replay_player_knowledge(
        {}, observations, RULES
    )

    assert len(result["facts"]) == 1
    assert result["facts"][0]["turn"] == 2


def test_knowledge_prompt_distinguishes_claims():
    knowledge = update_player_knowledge(
        initial_player_knowledge({}),
        1,
        "Ask Director Vale.",
        RULES,
    )

    prompt = format_player_knowledge(knowledge)

    assert "[REPORTED, turn 1]" in prompt
    assert "not verified facts" in prompt
