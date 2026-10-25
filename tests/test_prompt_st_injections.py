from rpbench.prompt_models import PromptMessage
from rpbench.prompt_presets import SillyTavernChatPreset
from rpbench.prompt_st_chat import render_st_chat_preset


def make_preset():
    return SillyTavernChatPreset({
        "prompts": [
            {
                "identifier": "main",
                "role": "system",
                "content": "Main instructions",
            },
            {
                "identifier": "note",
                "role": "system",
                "content": "In-chat note",
                "injection_position": 1,
                "injection_depth": 1,
                "injection_order": 100,
            },
            {
                "identifier": "chatHistory",
                "marker": True,
                "role": "system",
            },
            {
                "identifier": "post",
                "role": "system",
                "content": "Post-history",
            },
        ],
        "prompt_order": [
            {
                "character_id": 100000,
                "order": [
                    {"identifier": "main", "enabled": True},
                    {"identifier": "note", "enabled": True},
                    {"identifier": "chatHistory", "enabled": True},
                    {"identifier": "post", "enabled": True},
                ],
            },
        ],
    })


def test_injection_uses_history_depth():
    result = render_st_chat_preset(
        make_preset(),
        [
            PromptMessage("user", "First"),
            PromptMessage("assistant", "Second"),
            PromptMessage("user", "Third"),
        ],
        {},
        character_name="Elara",
        user_name="Rowan",
    )

    assert [message["content"] for message in result] == [
        "Main instructions",
        "First",
        "Second",
        "In-chat note",
        "Third",
        "Post-history",
    ]


def test_injection_ignores_manager_position():
    payload = make_preset().to_dict()

    order = payload["prompt_order"][0]["order"]
    note = next(
        item for item in order
        if item["identifier"] == "note"
    )

    order.remove(note)
    order.append(note)

    result = render_st_chat_preset(
        SillyTavernChatPreset(payload),
        [
            PromptMessage("user", "First"),
            PromptMessage("assistant", "Second"),
        ],
        {},
        character_name="Elara",
        user_name="Rowan",
    )

    assert [message["content"] for message in result] == [
        "Main instructions",
        "First",
        "In-chat note",
        "Second",
        "Post-history",
    ]


def test_disabled_injection_not_rendered():
    payload = make_preset().to_dict()

    for entry in payload["prompt_order"][0]["order"]:
        if entry["identifier"] == "note":
            entry["enabled"] = False

    result = render_st_chat_preset(
        SillyTavernChatPreset(payload),
        [PromptMessage("user", "Hello")],
        {},
        character_name="Elara",
        user_name="Rowan",
    )

    assert "In-chat note" not in [
        message["content"] for message in result
    ]
