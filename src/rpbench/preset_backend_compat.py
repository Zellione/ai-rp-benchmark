"""Explicit compatibility lowering for strict alternating chat templates."""

from copy import deepcopy
from hashlib import sha256
import json


def messages_sha256(messages):
    return sha256(
        json.dumps(
            messages,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def lower_to_alternating_chat(messages):
    """Lower initial system prompts and trailing post-history instructions.

    This is an explicit compatibility transformation, not an exact
    reproduction of SillyTavern's message serialization.
    """
    original = deepcopy(messages)

    if not original:
        raise ValueError("Empty prompt")

    first_user = next(
        (i for i, m in enumerate(original) if m["role"] == "user"),
        None,
    )

    if first_user is None:
        raise ValueError("No user turn")

    prefix = original[:first_user]
    conversation = original[first_user:]

    if any(m["role"] != "system" for m in prefix):
        raise ValueError("Unexpected pre-history role")

    system_content = "\n\n".join(
        m["content"] for m in prefix
    )

    trailing = []
    while conversation and conversation[-1]["role"] == "system":
        trailing.insert(0, conversation.pop())

    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Expected final user turn before post-history")

    if any(m["role"] not in ("user", "assistant") for m in conversation):
        raise ValueError("Unsupported in-history system injection")

    expected_role = "user"
    for message in conversation:
        if message["role"] != expected_role:
            raise ValueError("History roles are not alternating")
        expected_role = (
            "assistant" if expected_role == "user" else "user"
        )

    lowered = []

    if system_content:
        lowered.append({
            "role": "system",
            "content": system_content,
        })

    lowered.extend(deepcopy(conversation))

    if trailing:
        suffix = "\n\n".join(m["content"] for m in trailing)
        lowered[-1]["content"] += (
            "\n\n<rpbench-post-history-instructions>\n"
            + suffix
            + "\n</rpbench-post-history-instructions>"
        )

    metadata = {
        "strategy": "alternating-chat-v1",
        "original_messages_sha256": messages_sha256(original),
        "sent_messages_sha256": messages_sha256(lowered),
        "initial_system_messages_merged": len(prefix),
        "post_history_system_messages_relocated": len(trailing),
        "exact_sillytavern_serialization": False,
    }

    return lowered, metadata



def lower_to_alternating_chat_v2(messages):
    """Lower grouped user injections for strict alternating chat templates.

    Preserves chronological content order, but changes message boundaries
    and relocates trailing system instructions. This is not ST parity.
    """
    original = deepcopy(messages)

    if not original:
        raise ValueError("Empty prompt")

    first_user = next(
        (i for i, m in enumerate(original) if m["role"] == "user"),
        None,
    )
    if first_user is None:
        raise ValueError("No user turn")

    prefix = original[:first_user]
    conversation = deepcopy(original[first_user:])

    if any(m["role"] != "system" for m in prefix):
        raise ValueError("Unexpected pre-history role")

    trailing = []
    while conversation and conversation[-1]["role"] == "system":
        trailing.insert(0, conversation.pop())

    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Expected final user turn")

    if any(m["role"] not in {"user", "assistant"} for m in conversation):
        raise ValueError("Unsupported in-history system injection")

    merged = []
    user_merges = 0

    for message in conversation:
        if (
            merged
            and message["role"] == "user"
            and merged[-1]["role"] == "user"
        ):
            merged[-1]["content"] += (
                "\\n\\n" + message["content"]
            )
            user_merges += 1
        else:
            merged.append(dict(message))

    if any(
        a["role"] == b["role"]
        for a, b in zip(merged, merged[1:])
    ):
        raise ValueError(
            "Non-alternating history remains after user merging"
        )

    lowered = []
    if prefix:
        lowered.append({
            "role": "system",
            "content": "\\n\\n".join(m["content"] for m in prefix),
        })

    lowered.extend(merged)

    if trailing:
        lowered[-1]["content"] += (
            "\\n\\n<rpbench-post-history-instructions>"
            "\\n"
            + "\\n\\n".join(m["content"] for m in trailing)
            + "\\n</rpbench-post-history-instructions>"
        )

    from .prompt_chat import validate_strict_chat_template
    validate_strict_chat_template(lowered)

    metadata = {
        "strategy": "alternating-chat-v2",
        "original_messages_sha256": messages_sha256(original),
        "sent_messages_sha256": messages_sha256(lowered),
        "initial_system_messages_merged": len(prefix),
        "consecutive_user_messages_merged": user_merges,
        "post_history_system_messages_relocated": len(trailing),
        "exact_sillytavern_serialization": False,
    }

    return lowered, metadata
