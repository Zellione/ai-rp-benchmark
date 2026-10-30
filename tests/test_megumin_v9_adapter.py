from pathlib import Path

import pytest

from rpbench.megumin_v9_adapter import MeguminV9PresetAdapter
from rpbench.preset_adapter import render_with_adapter
from rpbench.prompt_models import PromptMessage


CONFIG = Path("presets/configs/megumin-v9-mirage.json")
ENGINE = Path("/scratch/rpbench-presets/megumin-v9-engine")


@pytest.mark.skipif(
    not CONFIG.exists() or not (ENGINE / "engine.json").exists(),
    reason="Historical local Megumin fixtures unavailable",
)
def test_megumin_v9_full_render():
    adapter = MeguminV9PresetAdapter(CONFIG, ENGINE)

    ordered = adapter.preset.ordered_prompts(100001)
    markers = {
        item["identifier"]: ""
        for item in ordered
        if item["enabled"]
        and item.get("marker", False)
        and item["identifier"] != "chatHistory"
    }
    markers["charDescription"] = "Mara Voss, Archivarin."

    history = [
        PromptMessage("user", "Hallo."),
        PromptMessage("assistant", "Mara nickt."),
        PromptMessage("user", "Wie geht es dir?"),
        PromptMessage("assistant", "Gut."),
        PromptMessage("user", "Erzähl mir mehr."),
    ]

    result = render_with_adapter(
        adapter,
        history,
        markers=markers,
        character_name="Mara Voss",
        user_name="Player",
        character_id=100001,
    )

    assert result.preset_id == "megumin-v9-mirage"
    assert len(result.messages) == 7
    assert result.messages[0]["role"] == "system"
    assert result.messages[-2]["role"] == "system"
    assert result.messages[-1]["role"] == "user"
    assert all(
        "[[" not in message["content"]
        for message in result.messages
    )
