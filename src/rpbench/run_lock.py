"""Linux process locks for individual RPBench run directories."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


class RunLockedError(RuntimeError):
    """Another process currently owns the run."""


@contextmanager
def run_lock(run_dir):
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    path = root / ".rpbench.lock"

    # Never delete or replace this file: doing so could create
    # two different inodes with independently held locks.
    fd = os.open(
        path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )

    try:
        try:
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RunLockedError(
                f"Run is already active: {root}"
            ) from exc

        yield
    finally:
        os.close(fd)
