"""Conservative recovery of interrupted JSONL appends."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


class JournalIntegrityError(ValueError):
    """Journal has corruption that cannot safely be recovered."""


def inspect_jsonl_tail(path):
    """Inspect a journal without changing it.

    Returns (action, valid_prefix_size, tail).
    """
    path = Path(path)
    if not path.exists():
        return "missing", 0, b""

    data = path.read_bytes()
    if not data:
        return "ok", 0, b""

    ends_with_newline = data.endswith(b"\n")
    lines = data.split(b"\n")

    if ends_with_newline:
        lines.pop()

    prefix = 0

    for index, raw in enumerate(lines):
        is_last = index == len(lines) - 1

        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            if is_last and not ends_with_newline:
                return "truncate", prefix, raw

            raise JournalIntegrityError(
                f"{path}: corrupt complete line {index + 1}"
            ) from exc

        if not isinstance(value, dict):
            raise JournalIntegrityError(
                f"{path}: line {index + 1} is not an object"
            )

        if not is_last or ends_with_newline:
            prefix += len(raw) + 1

    if ends_with_newline:
        return "ok", len(data), b""

    # A complete JSON object lacking only its terminating newline.
    return "add_newline", len(data), b""


def recover_jsonl_tail(path):
    """Recover only a partial final append, retaining its original bytes."""
    path = Path(path)
    action, prefix, tail = inspect_jsonl_tail(path)

    if action in {"missing", "ok"}:
        return action

    with path.open("r+b") as file:
        if action == "add_newline":
            file.seek(0, os.SEEK_END)
            file.write(b"\n")
        elif action == "truncate":
            if not tail:
                raise JournalIntegrityError(
                    "Refusing to truncate an empty tail"
                )

            backup_dir = path.parent / ".recovery"
            backup_dir.mkdir(mode=0o700, exist_ok=True)

            backup = (
                backup_dir /
                f"{path.name}.{uuid.uuid4().hex}.partial"
            )

            # Exclusive creation and fsync before touching the journal.
            with backup.open("xb") as saved:
                saved.write(tail)
                saved.flush()
                os.fsync(saved.fileno())

            directory_fd = os.open(
                backup_dir,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            file.truncate(prefix)
        else:
            raise AssertionError(action)

        file.flush()
        os.fsync(file.fileno())

    return action


def recover_run_journals(root):
    """Preflight all journals before recovering either one."""
    root = Path(root)
    paths = [
        root / "events.jsonl",
        root / "observations.jsonl",
    ]

    # Important: invalid complete lines stop recovery before any mutation.
    for path in paths:
        inspect_jsonl_tail(path)

    return {
        path.name: recover_jsonl_tail(path)
        for path in paths
    }
