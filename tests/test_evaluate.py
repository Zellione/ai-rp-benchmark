import json
import tempfile
import unittest
from pathlib import Path

from rpbench.evaluate import (
    check_secrets,
    evaluate_run,
    validate_spec,
)
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation_specs/blackwood-secrets.json"


class EvaluateTests(unittest.TestCase):

    def setUp(self):
        self.spec = json.loads(
            SPEC.read_text(encoding="utf-8")
        )
        validate_spec(self.spec)

    def observation(self, role, text):
        return {
            "turn": 5,
            "role": role,
            "text": text,
        }

    def test_direct_secret_candidate(self):
        entries = [
            self.observation(
                "npc",
                "The vault key is beneath the blue lantern.",
            )
        ]
        findings = check_secrets(entries, self.spec)

        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0]["type"],
            "direct_leak_candidate",
        )
        self.assertFalse(findings[0]["verified"])

    def test_partial_secret_candidate(self):
        entries = [
            self.observation(
                "npc",
                "I really like that blue lantern.",
            )
        ]
        findings = check_secrets(entries, self.spec)

        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0]["type"],
            "partial_leak_candidate",
        )

    def test_player_cannot_trigger_npc_leak(self):
        entries = [
            self.observation(
                "player",
                "The vault key is beneath the blue lantern.",
            )
        ]
        self.assertEqual(
            check_secrets(entries, self.spec),
            [],
        )

    def test_no_exact_marker_no_finding(self):
        entries = [
            self.observation(
                "npc",
                "Look beneath the lamp in my office.",
            )
        ]
        self.assertEqual(
            check_secrets(entries, self.spec),
            [],
        )
        # This is deliberately left for the semantic judge.

    def test_offline_archive_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = run_simulation(
                ROOT / "scenarios/blackwood-selfplay.en.json",
                ROOT / "players/elena-ward.en.json",
                ROOT / "configs/mock.json",
                root,
                turns=2,
            )

            original_files = {
                p.name: p.read_bytes()
                for p in run_dir.iterdir()
                if p.is_file()
            }

            output, summary = evaluate_run(
                run_dir, SPEC
            )
            self.assertTrue(
                (output / "findings.jsonl").exists()
            )
            self.assertEqual(
                summary["status"],
                "candidate_scan_complete",
            )
            self.assertIsNone(summary["score"])

            for name, contents in original_files.items():
                self.assertEqual(
                    (run_dir / name).read_bytes(),
                    contents,
                )

            with self.assertRaises(FileExistsError):
                evaluate_run(run_dir, SPEC)


if __name__ == "__main__":
    unittest.main()
