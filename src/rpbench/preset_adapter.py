"""Auditable, backend-independent preset adapters."""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .prompt_models import PromptMessage
from .prompt_presets import (
    DEFAULT_CHARACTER_ID,
    SillyTavernChatPreset,
)
from .prompt_st_chat import render_st_chat_preset
from .st_template_engine import (
    TemplateContext,
    render_template,
    repair_known_template_defect,
)


@dataclass(frozen=True)
class PresetContext:
    history: tuple[PromptMessage, ...]
    markers: Mapping[str, str]
    character_name: str
    user_name: str
    character_id: int | str = DEFAULT_CHARACTER_ID


@dataclass
class PresetResult:
    messages: list[dict[str, str]]
    preset_id: str
    source_sha256: str
    applied_hooks: list[str] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)


class PresetAdapter(Protocol):
    def render(self, context: PresetContext) -> PresetResult:
        ...


class IdentityPresetAdapter:
    """Preserve input messages without a preset intervention."""

    def render(self, context: PresetContext) -> PresetResult:
        return PresetResult(
            messages=[m.to_dict() for m in context.history],
            preset_id="baseline",
            source_sha256="none",
        )


class SillyTavernPresetAdapter:
    """Render the supported static SillyTavern Chat Completion subset."""

    def __init__(
        self,
        source: str | Path,
        *,
        preset_id: str,
        squash_system_messages: bool = False,
        allow_disabled_orphans: bool = False,
    ):
        self.source = Path(source)
        self.source_sha256 = sha256(
            self.source.read_bytes()
        ).hexdigest()
        self.preset = SillyTavernChatPreset.from_file(
            self.source,
            allow_disabled_orphans=allow_disabled_orphans,
        )
        self.preset_id = preset_id
        self.squash_system_messages = squash_system_messages

    def render(self, context: PresetContext) -> PresetResult:
        ordered = self.preset.ordered_prompts(
            context.character_id
        )

        unsupported = []

        for prompt in ordered:
            if not prompt["enabled"]:
                continue

            position = prompt.get("injection_position", 0)

            if type(position) is not int or position not in (0, 1):
                unsupported.append(
                    f"{prompt['identifier']}: "
                    f"injection_position={position}"
                )

        if unsupported:
            raise ValueError(
                "Unsupported preset features: "
                + ", ".join(unsupported)
            )

        template_context = TemplateContext(
            user_name=context.user_name,
            character_name=context.character_name,
        )
        repairs: list[str] = []

        def transform(identifier: str, content: str) -> str:
            repaired, applied = repair_known_template_defect(
                content,
                source_sha256=self.source_sha256,
                identifier=identifier,
            )
            repairs.extend(applied)
            return render_template(
                repaired,
                template_context,
                identifier=identifier,
            )

        messages = render_st_chat_preset(
            self.preset,
            context.history,
            context.markers,
            character_name=context.character_name,
            user_name=context.user_name,
            character_id=context.character_id,
            squash_system_messages=self.squash_system_messages,
            content_transform=transform,
        )

        return PresetResult(
            messages=messages,
            preset_id=self.preset_id,
            source_sha256=self.source_sha256,
            applied_hooks=[
                "sillytavern_chat_render",
                "st_template_subset",
                *repairs,
            ],
        )


def render_with_adapter(
    adapter: PresetAdapter,
    history: Sequence[PromptMessage],
    *,
    markers: Mapping[str, str],
    character_name: str,
    user_name: str,
    character_id: int | str = DEFAULT_CHARACTER_ID,
) -> PresetResult:
    context = PresetContext(
        history=tuple(history),
        markers=markers,
        character_name=character_name,
        user_name=user_name,
        character_id=character_id,
    )

    result = adapter.render(context)

    if not isinstance(result, PresetResult):
        raise TypeError("Adapter returned invalid result")

    if not result.messages:
        raise ValueError("Adapter produced no messages")

    for message in result.messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {
                "system", "user", "assistant"
            }
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError("Invalid rendered message")

    return result
