"""Lossless structural import of SillyTavern Chat Completion presets."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


DEFAULT_CHARACTER_ID = 100000


class SillyTavernChatPreset:
    """Preserve original preset JSON while exposing validated structure."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        allow_disabled_orphans: bool = False,
    ):
        if not isinstance(allow_disabled_orphans, bool):
            raise ValueError("allow_disabled_orphans must be boolean")
        self.allow_disabled_orphans = allow_disabled_orphans
        if not isinstance(payload, dict):
            raise ValueError("SillyTavern preset must be a JSON object")

        self._payload = deepcopy(payload)
        self._validate()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        allow_disabled_orphans: bool = False,
    ):
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(
            payload,
            allow_disabled_orphans=allow_disabled_orphans,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the original JSON structure."""
        return deepcopy(self._payload)

    def _validate(self) -> None:
        prompts = self._payload.get("prompts")
        orders = self._payload.get("prompt_order")

        if not isinstance(prompts, list):
            raise ValueError("Preset requires a prompts array")

        if not isinstance(orders, list):
            raise ValueError("Preset requires a prompt_order array")

        identifiers = set()

        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise ValueError("Prompt entries must be objects")

            identifier = prompt.get("identifier")

            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Prompt requires an identifier")

            if identifier in identifiers:
                raise ValueError(
                    f"Duplicate prompt identifier: {identifier}"
                )

            identifiers.add(identifier)

            if "content" in prompt:
                if not isinstance(prompt["content"], str):
                    raise ValueError(
                        f"Invalid content for {identifier}"
                    )

            if "role" in prompt:
                if prompt["role"] not in {
                    "system", "user", "assistant"
                }:
                    raise ValueError(
                        f"Unsupported role for {identifier}"
                    )

            if "marker" in prompt:
                if not isinstance(prompt["marker"], bool):
                    raise ValueError(
                        f"Invalid marker for {identifier}"
                    )

        character_ids = set()

        for group in orders:
            if not isinstance(group, dict):
                raise ValueError("Prompt order groups must be objects")

            character_id = group.get("character_id")
            order = group.get("order")

            if (
                not isinstance(character_id, (str, int))
                or isinstance(character_id, bool)
            ):
                raise ValueError("Invalid prompt order character_id")

            if character_id in character_ids:
                raise ValueError(
                    f"Duplicate prompt order group: {character_id}"
                )

            character_ids.add(character_id)

            if not isinstance(order, list):
                raise ValueError("Prompt order must be an array")

            seen = set()

            for entry in order:
                if not isinstance(entry, dict):
                    raise ValueError("Order entries must be objects")

                identifier = entry.get("identifier")
                enabled = entry.get("enabled")

                if identifier not in identifiers:
                    if not (
                        self.allow_disabled_orphans
                        and enabled is False
                    ):
                        raise ValueError(
                            f"Unknown ordered prompt: {identifier}"
                        )

                if identifier in seen:
                    raise ValueError(
                        f"Duplicate ordered prompt: {identifier}"
                    )

                if not isinstance(enabled, bool):
                    raise ValueError(
                        f"Invalid enabled value: {identifier}"
                    )

                seen.add(identifier)

    def prompt_definitions(self) -> list[dict[str, Any]]:
        return deepcopy(self._payload["prompts"])

    def ordered_prompts(
        self,
        character_id: str | int = DEFAULT_CHARACTER_ID,
    ) -> list[dict[str, Any]]:
        """Return prompt definitions with order-specific enabled flags.

        Marker prompts are retained. Nothing is rendered or expanded.
        """
        prompts = {
            item["identifier"]: item
            for item in self._payload["prompts"]
        }

        for group in self._payload["prompt_order"]:
            if group["character_id"] == character_id:
                result = []

                for entry in group["order"]:
                    identifier = entry["identifier"]

                    if identifier not in prompts:
                        if (
                            self.allow_disabled_orphans
                            and entry["enabled"] is False
                        ):
                            continue
                        raise ValueError(
                            f"Unknown ordered prompt: {identifier}"
                        )

                    item = deepcopy(prompts[identifier])
                    item["enabled"] = entry["enabled"]
                    result.append(item)

                return result

        raise ValueError(
            f"No prompt order for character_id={character_id}"
        )
