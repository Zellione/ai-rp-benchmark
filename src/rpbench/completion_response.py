"""Strict parsing of model-specific text-completion output."""

from __future__ import annotations


class CompletionParseError(ValueError):
    """A completion could not be separated into supported channels."""


SELF_HEADER = " to=self<|message|>"
USER_HEADER = "<|start|>assistant to=user<|message|>"

CONTROL_MARKERS = (
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
)


def parse_atem_completion(text: str) -> tuple[str, str]:
    """Return (visible_text, reasoning_text).

    Expected prompt ends with '<|start|>assistant'.

    Supported response:
      ' to=self<|message|>' + reasoning
      '<|start|>assistant to=user<|message|>' + visible

    Also supports an immediate ' to=user<|message|>' response.
    """
    if not isinstance(text, str) or not text:
        raise CompletionParseError("Empty ATEM completion")

    reasoning = ""

    if text.startswith(SELF_HEADER):
        remainder = text[len(SELF_HEADER):]

        if remainder.count(USER_HEADER) != 1:
            raise CompletionParseError(
                "Missing or repeated visible ATEM channel"
            )

        reasoning, visible = remainder.split(
            USER_HEADER, 1
        )

        # An explicit intermediate end marker is permitted.
        if reasoning.endswith("<|eom|>"):
            reasoning = reasoning[:-len("<|eom|>")]

    elif text.startswith(" to=user<|message|>"):
        visible = text[len(" to=user<|message|>"):]

    else:
        raise CompletionParseError(
            "Unexpected initial ATEM recipient"
        )

    # Some APIs include the final token while others stop
    # before returning it.
    if visible.endswith("<|eot|>"):
        visible = visible[:-len("<|eot|>")]

    if not visible.strip():
        raise CompletionParseError("Empty visible ATEM response")

    if any(marker in visible for marker in CONTROL_MARKERS):
        raise CompletionParseError(
            "Unexpected control marker in visible response"
        )

    # Muse can emit stray ATEM control tokens while thinking.
    # They are not part of the visible RP response.
    # Preserve the original generation in raw_response.
    for marker in CONTROL_MARKERS:
        reasoning = reasoning.replace(marker, "")

    return visible.strip(), reasoning.strip()


def parse_atem_direct_completion(text: str) -> tuple[str, str]:
    """Parse visible text after a pre-opened ATEM to=user header."""
    if not isinstance(text, str) or not text.strip():
        raise CompletionParseError("Empty direct ATEM completion")

    # The caller has already emitted:
    # <|start|>assistant to=user<|message|>
    # Therefore any further channel header is unexpected.
    if any(marker in text for marker in CONTROL_MARKERS):
        raise CompletionParseError(
            "Unexpected ATEM control marker in direct completion"
        )

    if "to=self<|message|>" in text or "to=user<|message|>" in text:
        raise CompletionParseError(
            "Unexpected ATEM recipient in direct completion"
        )

    return text.strip(), ""


def parse_atem_thinking_prefill(text: str) -> tuple[str, str]:
    """Parse completion after an already opened ATEM thinking channel.

    Prompt suffix:
        <|start|>assistant to=self<|message|>

    Generated continuation:
        reasoning
        <|start|>assistant to=user<|message|>
        visible roleplay
    """
    if not isinstance(text, str) or not text:
        raise CompletionParseError(
            "Empty ATEM thinking-prefill completion"
        )

    if text.count(USER_HEADER) != 1:
        raise CompletionParseError(
            "Expected exactly one visible ATEM channel"
        )

    reasoning, visible = text.split(USER_HEADER, 1)

    if reasoning.endswith("<|eom|>"):
        reasoning = reasoning[:-len("<|eom|>")]

    if visible.endswith("<|eot|>"):
        visible = visible[:-len("<|eot|>")]

    if not visible.strip():
        raise CompletionParseError(
            "Empty visible ATEM response"
        )

    if any(marker in visible for marker in CONTROL_MARKERS):
        raise CompletionParseError(
            "Unexpected control marker in visible response"
        )

    # Muse can emit stray ATEM control tokens while thinking.
    # They are not part of the visible RP response.
    # Preserve the original generation in raw_response.
    for marker in CONTROL_MARKERS:
        reasoning = reasoning.replace(marker, "")

    return visible.strip(), reasoning.strip()
