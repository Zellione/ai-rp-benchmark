"""Explicit compatibility lowering for pre-final-user system injections."""

from copy import deepcopy


def relocate_prefinal_system(messages):
    """Move system instructions immediately before the final user turn.

    This changes instruction roles and is not exact ST serialization.
    """
    result = deepcopy(messages)

    if len(result) < 2 or result[-1]["role"] != "user":
        raise ValueError("Expected a final user message")

    start = len(result) - 2
    while start >= 0 and result[start]["role"] == "system":
        start -= 1

    first = start + 1
    if first == len(result) - 1:
        return result, 0

    if first == 0:
        # A leading system prefix is valid and must retain its role.
        return result, 0

    instructions = result[first:-1]
    if any(not m["content"].strip() for m in instructions):
        raise ValueError("Empty system injection")

    # Preserve each instruction's original text in the sent payload.
    suffix = "\n\n".join(m["content"] for m in instructions)
    final_user = result[-1].copy()
    final_user["content"] = (
        "<rpbench-prefinal-system-instructions>\n"
        + suffix
        + "\n</rpbench-prefinal-system-instructions>\n\n"
        + final_user["content"]
    )

    return [*result[:first], final_user], len(instructions)
