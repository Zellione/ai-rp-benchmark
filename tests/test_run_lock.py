import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rpbench.run_lock import RunLockedError, run_lock


class RunLockTests(unittest.TestCase):

    def test_same_process_cannot_acquire_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            with run_lock(temp):
                with self.assertRaises(RunLockedError):
                    with run_lock(temp):
                        pass

    def test_lock_released_after_context_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            with run_lock(temp):
                pass
            with run_lock(temp):
                pass

    def test_lock_released_after_process_kill(self):
        with tempfile.TemporaryDirectory() as temp:
            code = """
import sys
import time
from rpbench.run_lock import run_lock
with run_lock(sys.argv[1]):
    print("LOCKED", flush=True)
    time.sleep(120)
"""
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    temp,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                self.assertEqual(
                    proc.stdout.readline().strip(),
                    "LOCKED",
                )

                with self.assertRaises(RunLockedError):
                    with run_lock(temp):
                        pass

                proc.kill()
                proc.wait(timeout=10)

                # Kernel releases the lock, even after SIGKILL.
                with run_lock(temp):
                    pass
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)

    def test_lock_file_is_persistent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".rpbench.lock"

            with run_lock(temp):
                self.assertTrue(path.is_file())

            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
