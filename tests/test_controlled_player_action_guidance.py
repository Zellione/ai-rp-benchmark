from rpbench.player_controller import (
    build_controlled_player_instruction,
    render_controlled_player_request,
)


def state():
    return {
        "goals": {
            "subgoal-01": {
                "description": "Request access to the register",
                "status": "pending",
            },
        },
    }


def test_controller_encourages_concrete_action():
    text = build_controlled_player_instruction(state())

    assert "one concrete next action" in text
    assert "do so now" in text
    assert "Do not repeatedly ask" in text
    assert "Never assume NPC cooperation" in text
    assert "without rushing" not in text


def test_rendering_preserves_original_messages():
    messages = [
        {"role": "system", "content": "You are Alex."},
        {"role": "user", "content": "Mara offers the register."},
    ]

    original = [dict(item) for item in messages]

    rendered = render_controlled_player_request(
        messages, state()
    )

    assert messages == original
    assert "one concrete next action" in rendered[0]["content"]
    assert rendered[1] == messages[1]
