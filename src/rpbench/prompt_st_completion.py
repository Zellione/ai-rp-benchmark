"""Minimal SillyTavern Text Completion preset imports.

Context and Instruct presets are kept separate, as they are
in SillyTavern. No rendering or backend integration yet.
"""

from copy import deepcopy
import json
import re
from pathlib import Path


class SillyTavernContextPreset:
    def __init__(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Context preset must be an object")

        value = payload.get("story_string")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Context preset requires story_string")

        position = payload.get("story_string_position", 0)
        depth = payload.get("story_string_depth", 1)
        role = payload.get("story_string_role", 0)

        if type(position) is not int or position not in (0, 1):
            raise ValueError("Invalid story_string_position")

        if type(depth) is not int or depth < 0:
            raise ValueError("Invalid story_string_depth")

        if type(role) is not int or role not in (0, 1, 2):
            raise ValueError("Invalid story_string_role")

        self._payload = deepcopy(payload)

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(
            Path(path).read_text(encoding="utf-8")
        ))

    def to_dict(self):
        return deepcopy(self._payload)


class SillyTavernInstructPreset:
    def __init__(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Instruct preset must be an object")

        fields = (
            "input_sequence",
            "output_sequence",
            "input_suffix",
            "output_suffix",
            "system_sequence",
            "system_suffix",
            "story_string_prefix",
            "story_string_suffix",
            "stop_sequence",
        )

        for field in fields:
            value = payload.get(field, "")
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid instruct preset field: {field}"
                )

        for field in (
            "wrap",
            "macro",
            "system_same_as_user",
            "sequences_as_stop_strings",
        ):
            if field in payload and type(payload[field]) is not bool:
                raise ValueError(
                    f"Invalid instruct preset field: {field}"
                )

        self._payload = deepcopy(payload)

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(
            Path(path).read_text(encoding="utf-8")
        ))

    def to_dict(self):
        return deepcopy(self._payload)


_ST_TOKEN = re.compile(r"{{(.*?)}}", re.DOTALL)
_ST_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def render_st_story_string(
    preset: SillyTavernContextPreset,
    values: dict[str, str],
) -> str:
    """Render a small, explicit SillyTavern story-string subset.

    Supported:
    - {{field}}
    - {{#if field}}...{{/if}}
    - {{trim}}

    Nested conditionals and other Handlebars constructs are not
    supported. Unsupported syntax fails explicitly.
    """
    if not isinstance(values, dict):
        raise ValueError("Story values must be an object")

    for key, value in values.items():
        if (
            not isinstance(key, str)
            or not _ST_NAME.fullmatch(key)
            or not isinstance(value, str)
        ):
            raise ValueError("Invalid story-string value")

    template = preset.to_dict()["story_string"]

    result = []
    active = True
    inside_if = False
    position = 0

    for match in _ST_TOKEN.finditer(template):
        literal = template[position:match.start()]
        if active:
            result.append(literal)

        tag = match.group(1).strip()

        if tag.startswith("#if "):
            name = tag[4:].strip()
            if inside_if or not _ST_NAME.fullmatch(name):
                raise ValueError("Unsupported nested/invalid if")

            inside_if = True
            active = bool(values.get(name, ""))

        elif tag == "/if":
            if not inside_if:
                raise ValueError("Unexpected /if")

            inside_if = False
            active = True

        elif tag == "trim":
            if active:
                result.append("{{trim}}")

        elif _ST_NAME.fullmatch(tag):
            if active:
                result.append(values.get(tag, ""))

        else:
            raise ValueError(
                f"Unsupported story-string tag: {tag}"
            )

        position = match.end()

    if inside_if:
        raise ValueError("Unclosed story-string if")

    if active:
        result.append(template[position:])

    output = "".join(result)

    # SillyTavern removes trim and adjacent line breaks
    # after template expansion.
    output = re.sub(
        r"(?:\r?\n)*{{trim}}(?:\r?\n)*",
        "",
        output,
        flags=re.IGNORECASE,
    )

    if "{{" in output or "}}" in output:
        raise ValueError("Unresolved story-string syntax")

    return output.lstrip("\n")



def format_st_instruct_message(
    preset: SillyTavernInstructPreset,
    role: str,
    content: str,
    *,
    name: str = "",
) -> str:
    """Format a standard text-only SillyTavern instruct message.

    Scope: ordinary messages, no group-specific name behavior,
    no first/last sequence overrides or external macros.
    """
    if role not in {"system", "user", "assistant"}:
        raise ValueError("Unsupported instruct role")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Instruct content must be nonempty")

    if not isinstance(name, str):
        raise ValueError("Instruct name must be text")

    settings = preset.to_dict()
    wrap = settings.get("wrap", False)
    macro = settings.get("macro", False)

    if role == "system":
        if settings.get("system_same_as_user", False):
            prefix = settings.get("input_sequence", "")
            suffix = settings.get("input_suffix", "")
        else:
            prefix = settings.get("system_sequence", "")
            suffix = settings.get("system_suffix", "")
    elif role == "user":
        prefix = settings.get("input_sequence", "")
        suffix = settings.get("input_suffix", "")
    else:
        prefix = settings.get("output_sequence", "")
        suffix = settings.get("output_suffix", "")

    if macro:
        speaker = name or "System"
        prefix = re.sub(
            r"{{name}}",
            lambda _: speaker,
            prefix,
            flags=re.IGNORECASE,
        )
        suffix = re.sub(
            r"{{name}}",
            lambda _: speaker,
            suffix,
            flags=re.IGNORECASE,
        )

    # SillyTavern's default "force" behavior only adds
    # names in specific group/forced-avatar situations.
    # Those situations are outside this initial subset.
    include_name = (
        role != "system"
        and settings.get("names_behavior") == "always"
        and bool(name)
    )

    if not suffix and wrap:
        suffix = "\n"

    separator = "\n" if wrap else ""
    body = (
        f"{name}: {content}" if include_name else content
    ) + suffix

    return separator.join(
        part for part in (prefix, body) if part
    )



def format_st_generation_prefix(
    preset: SillyTavernInstructPreset,
    *,
    name: str = "",
) -> str:
    """Format SillyTavern's normal NPC generation boundary.

    Subset of formatInstructModePrompt:
    - Normal AI response only
    - No impersonation, quiet generation or prompt bias
    - No group-specific FORCE name behavior
    """
    if not isinstance(name, str):
        raise ValueError("Generation name must be text")

    settings = preset.to_dict()
    sequence = (
        settings.get("last_output_sequence")
        or settings.get("output_sequence", "")
    )

    if settings.get("macro", False):
        sequence = re.sub(
            r"{{name}}",
            lambda _: name or "System",
            sequence,
            flags=re.IGNORECASE,
        )

    wrap = settings.get("wrap", False)
    separator = "\n" if wrap else ""
    include_name = (
        bool(name)
        and settings.get("names_behavior") == "always"
    )

    if include_name:
        name_filler = ""
        last = settings.get("last_output_sequence", "")
        normal = settings.get("output_sequence", "")

        if (
            last
            and normal
            and sequence != normal
            and normal[-1].isspace()
            and not last[-1].isspace()
        ):
            name_filler = normal[-1]

        result = (
            separator
            + sequence
            + separator
            + name_filler
            + f"{name}:"
        )
    else:
        result = separator + sequence

    if wrap:
        result = result.rstrip()

    return result + ("" if include_name else separator)



def render_st_completion_prompt(
    messages: list[dict[str, str]],
    fmt: dict,
) -> str:
    """Build a text-only RP prompt from ST context/instruct presets.

    Supported scope:
    - Story String in relative context position
    - Ordinary text chat messages
    - Normal assistant generation
    - Explicit speaker names
    - No examples, extensions or context trimming
    """
    if not isinstance(fmt, dict):
        raise ValueError("Completion format must be an object")

    if not isinstance(messages, list) or not messages:
        raise ValueError("Completion messages must be nonempty")

    if messages[0].get("role") != "system":
        raise ValueError("First message must contain NPC context")

    context = SillyTavernContextPreset(
        fmt["context_preset"]
    )
    instruct = SillyTavernInstructPreset(
        fmt["instruct_preset"]
    )

    context_settings = context.to_dict()
    if context_settings.get("story_string_position", 0) != 0:
        raise ValueError(
            "In-chat story-string placement is not supported"
        )

    user_name = fmt["user_name"]
    character_name = fmt["character_name"]

    if any(
        not isinstance(value, str) or not value.strip()
        for value in (user_name, character_name)
    ):
        raise ValueError("Invalid completion speaker names")

    values = {
        "system": messages[0]["content"],
        "user": user_name,
        "char": character_name,
    }
    values.update(fmt.get("story_values", {}))

    if values["system"] != messages[0]["content"]:
        raise ValueError(
            "story_values must not override NPC system context"
        )

    story = render_st_story_string(context, values)

    instruct_settings = instruct.to_dict()
    wrap = instruct_settings.get("wrap", False)

    # Corresponds to the supported relative story-string path.
    if story:
        prefix = instruct_settings.get(
            "story_string_prefix", ""
        )
        suffix = instruct_settings.get(
            "story_string_suffix", ""
        )

        if prefix:
            story = prefix + ("\n" if wrap else "") + story
        if suffix:
            story += suffix

    parts = [story] if story else []

    chat_start = context_settings.get("chat_start", "")
    if chat_start:
        if not isinstance(chat_start, str):
            raise ValueError("Invalid chat_start")
        parts.append(chat_start)

    for message in messages[1:]:
        if not isinstance(message, dict):
            raise ValueError("Invalid completion message")

        if set(message) != {"role", "content"}:
            raise ValueError("Unsupported completion message fields")

        role = message["role"]
        name = (
            user_name if role == "user"
            else character_name if role == "assistant"
            else ""
        )

        parts.append(format_st_instruct_message(
            instruct,
            role,
            message["content"],
            name=name,
        ))

    parts.append(format_st_generation_prefix(
        instruct,
        name=character_name,
    ))

    # Instruct sequences already own their whitespace.
    # No generic separator is inserted between messages.
    return "".join(parts)
