import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rpbench.core import append, dump, load
from rpbench.resume import resume_simulation
from rpbench.state import make_observation


class SelfplayResumeTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.manifest = {
            "schema_version": 2,
            "type": "goal_driven_selfplay",
            "status": "failed",
            "planned_turns": 3,
            "completed_turns": 0,
            "scenario": {
                "id": "resume-test",
                "title": "Resume Test",
                "opening": "Begin.",
            },
            "player": {"name": "Tester"},
            "backend": {
                "name": "mock",
                "model": "mock",
                "mode": "chat",
            },
            "sampling": {},
            "player_sampling": {},
            "prompt_format": {},
            "player_prompt_format": {},
            "rendered_player_system_prompt": "Play Tester.",
            "rendered_npc_system_prompt": "Play NPC.",
        }

        dump(self.root / "manifest.json", self.manifest)

        (self.root / "conversation.md").write_text(
            "# Resume Test\n\n"
            "Player: Tester\n\n"
            "Mode: Goal-driven self-play\n\n",
            encoding="utf-8",
        )

    def observations(self):
        file = self.root / "observations.jsonl"
        if not file.exists():
            return []

        import json
        return [
            json.loads(line)
            for line in file.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def seed_pair(self):
        player = "Already archived player"
        npc = "Already archived NPC"

        append(
            self.root / "observations.jsonl",
            make_observation(1, "player", player),
        )
        append(
            self.root / "observations.jsonl",
            make_observation(1, "npc", npc),
        )

        with (self.root / "conversation.md").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(
                "## Turn 1\n\n"
                "### Player (Tester)\n\n"
                f"{player}\n\n"
                "### NPC\n\n"
                f"{npc}\n\n"
            )

        self.manifest["completed_turns"] = 1
        dump(self.root / "manifest.json", self.manifest)

    def test_resume_after_complete_turn(self):
        self.seed_pair()
        before = (
            self.root / "conversation.md"
        ).read_bytes()

        resume_simulation(self.root)

        result = load(self.root / "manifest.json")
        entries = self.observations()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_turns"], 3)
        self.assertEqual(len(entries), 6)
        self.assertEqual(
            entries[0]["text"],
            "Already archived player",
        )
        self.assertTrue(
            (self.root / "conversation.md")
            .read_bytes().startswith(before)
        )

    def test_resume_after_player_response(self):
        self.seed_pair()

        append(
            self.root / "observations.jsonl",
            make_observation(
                2, "player", "Preserved partial player"
            ),
        )

        resume_simulation(self.root)
        entries = self.observations()

        self.assertEqual(len(entries), 6)
        self.assertEqual(
            entries[2]["text"],
            "Preserved partial player",
        )
        self.assertEqual(entries[3]["role"], "npc")

    def test_recover_response_before_observation(self):
        self.seed_pair()

        append(
            self.root / "events.jsonl",
            {
                "type": "response",
                "turn": 2,
                "role": "player",
                "text": "Journaled player answer",
                "response_status": "complete",
            },
        )

        resume_simulation(self.root)
        entries = self.observations()

        self.assertEqual(
            entries[2]["text"],
            "Journaled player answer",
        )
        self.assertEqual(len(entries), 6)

    def test_reject_modified_transcript(self):
        self.seed_pair()

        with (self.root / "conversation.md").open(
            "a", encoding="utf-8"
        ) as f:
            f.write("UNEXPECTED CONTENT")

        with self.assertRaisesRegex(
            ValueError, "Transcript differs"
        ):
            resume_simulation(self.root)

    def test_resume_orphaned_running_manifest(self):
        self.seed_pair()

        self.manifest["status"] = "running"
        dump(self.root / "manifest.json", self.manifest)

        resume_simulation(self.root)

        result = load(self.root / "manifest.json")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_turns"], 3)
        self.assertEqual(len(self.observations()), 6)

    def test_recover_after_sigkill_mid_journal_append(self):
        self.seed_pair()
        self.manifest["status"] = "running"
        dump(self.root / "manifest.json", self.manifest)

        code = """
import os
import sys
import time
from pathlib import Path
from rpbench.run_lock import run_lock

root = Path(sys.argv[1])
with run_lock(root):
    with (root / "events.jsonl").open("ab") as file:
        file.write(
            b'{"type":"response","turn":2,"role":"player"'
        )
        file.flush()
        os.fsync(file.fileno())
    print("PARTIAL_WRITTEN", flush=True)
    time.sleep(120)
"""

        process = subprocess.Popen(
            [sys.executable, "-c", code, str(self.root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            self.assertEqual(
                process.stdout.readline().strip(),
                "PARTIAL_WRITTEN",
            )
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

        resume_simulation(self.root)

        manifest = load(self.root / "manifest.json")
        entries = self.observations()

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_turns"], 3)
        self.assertEqual(len(entries), 6)
        self.assertEqual(
            entries[0]["text"],
            "Already archived player",
        )

        backups = list(
            (self.root / ".recovery").glob(
                "events.jsonl.*.partial"
            )
        )
        self.assertEqual(len(backups), 1)

    def test_extend_existing_run(self):
        self.seed_pair()

        resume_simulation(self.root, turns=5)
        result = load(self.root / "manifest.json")

        self.assertEqual(result["planned_turns"], 5)
        self.assertEqual(result["completed_turns"], 5)
        self.assertEqual(len(self.observations()), 10)


if __name__ == "__main__":
    unittest.main()
