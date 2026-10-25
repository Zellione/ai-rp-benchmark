"""Backend-independent in-chat prompt injection primitives."""

from dataclasses import dataclass
from typing import Sequence

from .prompt_models import PromptMessage, Role


@dataclass(frozen=True)
class ChatInjection:
    identifier: str
    role: Role
    content: str
    depth: int
    order: int = 100

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("Injection requires an identifier")

        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported injection role")

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Injection content must be nonempty")

        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("Injection depth must be nonnegative")

        if type(self.order) is not int:
            raise ValueError("Injection order must be an integer")


def apply_chat_injections(
    history: Sequence[PromptMessage],
    injections: Sequence[ChatInjection],
) -> list[PromptMessage]:
    """Insert grouped injections relative to original chat history.

    Depth 0 is after the last message.
    Equal-depth injections are grouped by role and ordered by
    injection order, preserving input order for ties.

    Joined content uses an RPBench-defined newline separator;
    exact SillyTavern payload equivalence is not yet established.
    """
    original = list(history)
    buckets: dict[int, list[tuple[int, ChatInjection]]] = {}

    for index, injection in enumerate(injections):
        position = max(0, len(original) - injection.depth)
        buckets.setdefault(position, []).append((index, injection))

    role_order = {
        "user": 0,
        "assistant": 1,
        "system": 2,
    }

    result = []

    for position in range(len(original) + 1):
        selected = buckets.get(position, [])

        for role in ("user", "assistant", "system"):
            same_role = sorted(
                (
                    (index, injection)
                    for index, injection in selected
                    if injection.role == role
                ),
                key=lambda pair: (
                    pair[1].order,
                    pair[0],
                ),
            )

            if same_role:
                content = "\n".join(
                    injection.content
                    for _, injection in same_role
                )
                result.append(PromptMessage(role, content))

        if position < len(original):
            result.append(original[position])

    return result
