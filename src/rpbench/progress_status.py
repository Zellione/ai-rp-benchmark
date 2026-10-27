"""Lightweight live progress reporting for RPBench."""

from datetime import datetime
import sys


def emit(message):
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [RPBench] {message}",
        file=sys.stderr,
        flush=True,
    )
