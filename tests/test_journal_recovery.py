import json
import tempfile
import unittest
from pathlib import Path

from rpbench.core import dump, load
from rpbench.journal_recovery import (
    JournalIntegrityError,
    recover_jsonl_tail,
    recover_run_journals,
)


class AtomicDumpTests(unittest.TestCase):

    def test_atomic_dump_replaces_existing_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"

            dump(path, {"status": "running"})
            dump(path, {"status": "complete", "turns": 20})

            self.assertEqual(
                load(path),
                {"status": "complete", "turns": 20},
            )
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(
                list(Path(temp).glob("*.tmp")),
                [],
            )


class JournalRecoveryTests(unittest.TestCase):

    def test_truncated_tail_preserves_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            prefix = b'{"type":"response","turn":1}\n'
            tail = b'{"type":"response","turn":2,"text":"incomp'

            path.write_bytes(prefix + tail)

            self.assertEqual(
                recover_jsonl_tail(path),
                "truncate",
            )
            self.assertEqual(path.read_bytes(), prefix)

            backups = list(
                (Path(temp) / ".recovery").glob("*.partial")
            )
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), tail)

    def test_valid_tail_without_newline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_bytes(b'{"turn":1}')

            self.assertEqual(
                recover_jsonl_tail(path),
                "add_newline",
            )
            self.assertEqual(
                path.read_bytes(),
                b'{"turn":1}\n',
            )

    def test_corrupt_complete_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            original = (
                b'{"turn":1}\n'
                b'{"turn":BROKEN}\n'
                b'{"turn":3'
            )
            path.write_bytes(original)

            with self.assertRaises(JournalIntegrityError):
                recover_jsonl_tail(path)

            self.assertEqual(path.read_bytes(), original)

    def test_preflight_prevents_partial_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = root / "events.jsonl"
            observations = root / "observations.jsonl"

            events.write_bytes(
                b'{"turn":1}\n{"broken":'
            )
            observations.write_bytes(
                b'{"turn":1}\nnot-json\n'
            )

            before = events.read_bytes()

            with self.assertRaises(JournalIntegrityError):
                recover_run_journals(root)

            self.assertEqual(events.read_bytes(), before)

    def test_recovery_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_bytes(b'{"turn":1}\n{"turn":')

            self.assertEqual(
                recover_jsonl_tail(path),
                "truncate",
            )
            self.assertEqual(recover_jsonl_tail(path), "ok")

            backups = list(
                (Path(temp) / ".recovery").glob("*.partial")
            )
            self.assertEqual(len(backups), 1)


if __name__ == "__main__":
    unittest.main()
