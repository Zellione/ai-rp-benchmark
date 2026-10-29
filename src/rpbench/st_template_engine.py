"""Small, non-executable subset of SillyTavern template expressions.

This is a compatibility interpreter, not a JavaScript runtime.
Unsupported expressions fail closed.
"""

import re
from dataclasses import dataclass, field


TOKEN = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
VARIABLE = r"\.[A-Za-z_][A-Za-z_0-9]*"
ASSIGNMENT = re.compile(
    rf"^({VARIABLE})\s*=\s*([01])$"
)
CONDITION = re.compile(
    rf"!?{VARIABLE}(?:\s*\|\|\s*!?{VARIABLE})*"
)


class TemplateError(ValueError):
    pass


@dataclass
class TemplateContext:
    variables: dict[str, bool] = field(default_factory=dict)
    user_name: str = ""
    character_name: str = ""


def evaluate_condition(expression, variables):
    expression = expression.strip()

    if not CONDITION.fullmatch(expression):
        raise TemplateError(
            f"Unsupported condition: {expression!r}"
        )

    values = []

    for term in expression.split("||"):
        term = term.strip()
        negate = term.startswith("!")
        name = term[1:] if negate else term

        if name not in variables:
            raise TemplateError(
                f"Undefined template variable: {name}"
            )

        value = variables[name]
        values.append(not value if negate else value)

    return any(values)


def render_template(
    content: str,
    context: TemplateContext,
    *,
    identifier: str = "<unknown>",
) -> str:
    if not isinstance(content, str):
        raise TemplateError("Template content must be text")

    output = []
    frames = []
    active = True
    cursor = 0

    try:
        for match in TOKEN.finditer(content):
            if active:
                output.append(content[cursor:match.start()])

            expression = match.group(1).strip()
            cursor = match.end()

            if expression.startswith("//"):
                # SillyTavern-style inline template comment.
                continue

            if expression.startswith("if "):
                condition = evaluate_condition(
                    expression[3:],
                    context.variables,
                ) if active else False

                frames.append({
                    "parent": active,
                    "condition": condition,
                    "else_seen": False,
                })
                active = active and condition
                continue

            if expression == "else":
                if not frames or frames[-1]["else_seen"]:
                    raise TemplateError("Unexpected else")

                frame = frames[-1]
                frame["else_seen"] = True
                active = (
                    frame["parent"]
                    and not frame["condition"]
                )
                continue

            if expression == "/if":
                if not frames:
                    raise TemplateError("Unexpected /if")
                active = frames.pop()["parent"]
                continue

            if not active:
                continue

            assignment = ASSIGNMENT.fullmatch(expression)

            if assignment:
                name, value = assignment.groups()
                context.variables[name] = value == "1"
                continue

            if expression.lower() == "trim":
                if output:
                    output[-1] = output[-1].rstrip()
                continue

            if expression.lower() == "user":
                if not context.user_name:
                    raise TemplateError("Missing user name")
                output.append(context.user_name)
                continue

            if expression.lower() in {"char", "group"}:
                # SillyTavern {{group}} resolves to the character
                # name in solo chats. RPBench currently supports
                # only the solo-chat rendering context.
                if not context.character_name:
                    raise TemplateError("Missing character name")
                output.append(context.character_name)
                continue

            raise TemplateError(
                f"Unsupported template expression: {expression!r}"
            )

        if frames:
            raise TemplateError("Unclosed if block")

        if active:
            output.append(content[cursor:])

        rendered = "".join(output)

        if "{{" in rendered or "}}" in rendered:
            raise TemplateError(
                "Unprocessed template delimiters remain"
            )

        return rendered

    except TemplateError as error:
        raise TemplateError(
            f"{identifier}: {error}"
        ) from error


VOYAGE_V6_SHA256 = (
    "8df92f83c3bc2e40ef53344f7a43fbe0"
    "b967c0fb446f94b77e18928cffd23210"
)
VOYAGE_LOCATION_ID = "8b3ee347-7cbe-4744-898f-b67a2241ba59"

VOYAGE_BAD_TOKEN = "{{ trim }{{ if .setting_tracker_location }}"
VOYAGE_FIXED_TOKEN = "{{ trim }}{{ if .setting_tracker_location }}"


def repair_known_template_defect(
    content: str,
    *,
    source_sha256: str,
    identifier: str,
) -> tuple[str, list[str]]:
    """Apply an explicitly identified source-specific compatibility fix."""
    if (
        source_sha256 != VOYAGE_V6_SHA256
        or identifier != VOYAGE_LOCATION_ID
    ):
        return content, []

    count = content.count(VOYAGE_BAD_TOKEN)

    if count != 1:
        raise TemplateError(
            "Voyage v6 location-sheet defect did not match exactly"
        )

    return (
        content.replace(
            VOYAGE_BAD_TOKEN,
            VOYAGE_FIXED_TOKEN,
            1,
        ),
        ["voyage-v6-location-sheet-missing-brace"],
    )
