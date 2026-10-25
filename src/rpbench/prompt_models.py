"""Validated, backend-independent prompt structures."""

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class PromptMessage:
    role: Role
    content: str

    def __post_init__(self):
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {self.role}")

        if not isinstance(self.content, str):
            raise ValueError("Message content must be a string")

        if not self.content.strip():
            raise ValueError("Message content must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True)
class PromptSection:
    id: str
    role: Role
    content: str
    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Prompt section requires an ID")

        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported section role: {self.role}")

        if not isinstance(self.content, str):
            raise ValueError("Section content must be a string")

        if not isinstance(self.enabled, bool):
            raise ValueError("Section enabled must be boolean")

        if self.enabled and not self.content.strip():
            raise ValueError("Enabled section must have content")
