"""Historical Megumin V9 Mirage adapter for RPBench."""

import hashlib
import json
from pathlib import Path

from .megumin_v9 import make_profile_resolver
from .preset_adapter import PresetResult
from .prompt_presets import SillyTavernChatPreset
from .prompt_st_chat import render_st_chat_preset
from .st_template_engine import TemplateContext, render_template


class MeguminV9PresetAdapter:
    def __init__(self, config_path, engine_directory):
        self.config_path = Path(config_path)
        self.engine_directory = Path(engine_directory)

        self.config = json.loads(
            self.config_path.read_text(encoding="utf-8")
        )

        self.source = Path(self.config["source"]["preset"])
        self.source_sha256 = hashlib.sha256(
            self.source.read_bytes()
        ).hexdigest()

        if self.source_sha256 != self.config["source"]["preset_sha256"]:
            raise ValueError("Megumin source fingerprint mismatch")

        self.preset = SillyTavernChatPreset.from_file(
            self.source,
            allow_disabled_orphans=True,
        )

        self.engine = json.loads(
            (self.engine_directory / "engine.json").read_text(
                encoding="utf-8"
            )
        )
        self.cot = json.loads(
            (self.engine_directory / "cot.json").read_text(
                encoding="utf-8"
            )
        )

        self.resolver = make_profile_resolver(
            self.engine, self.cot, self.config
        )

        self.preset_id = "megumin-v9-mirage"

    def render(self, context):
        if context.character_id != self.config["character_id"]:
            raise ValueError("Megumin character ID mismatch")

        template_context = TemplateContext(
            user_name=context.user_name,
            character_name=context.character_name,
        )

        def transform(identifier, content):
            expanded = self.resolver.render(
                content, identifier=identifier
            )
            return render_template(
                expanded,
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
            squash_system_messages=bool(
                self.preset.to_dict().get(
                    "squash_system_messages", False
                )
            ),
            content_transform=transform,
        )

        return PresetResult(
            messages=messages,
            preset_id=self.preset_id,
            source_sha256=self.source_sha256,
            applied_hooks=[
                "megumin_v9_historical_engine",
                "megumin_v9_profile_resolver",
                "sillytavern_chat_render",
                "st_template_subset",
            ],
            unsupported_features=[
                "dynamic_npc_bank",
                "extension_memory",
                "story_planner",
                "image_generation",
                "assistant_prefill",
            ],
        )
