import json
import tempfile
import unittest
from pathlib import Path

from rpbench.core import run
from rpbench.evaluate import evaluate_run
from rpbench.observations import (
    read_observations,
    source_path,
)
from rpbench.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation_specs/blackwood-secrets.json"


class ObservationReaderTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def legacy_run(self):
        return run(
            ROOT / "scenarios/suspicious-archivist.json",
            ROOT / "configs/mock.json",
            str(self.root),
        )

    def selfplay_run(self):
        return run_simulation(
            ROOT / "scenarios/blackwood-selfplay.en.json",
            ROOT / "players/elena-ward.en.json",
            ROOT / "configs/mock.json",
            self.root,
            turns=3,
        )

    def test_legacy_fixed_scenario(self):
        archive = self.legacy_run()
        entries = read_observations(archive)

        self.assertEqual(len(entries), 6)
        self.assertEqual(
            [e["role"] for e in entries],
            ["player", "npc"] * 3,
        )
        self.assertIn(
            "missing key", entries[0]["text"]
        )
        self.assertIn(
            "mock answer for turn 3",
            entries[-1]["text"],
        )

    def test_selfplay(self):
        archive = self.selfplay_run()
        entries = read_observations(archive)

        self.assertEqual(len(entries), 6)
        self.assertEqual(
            source_path(archive).name,
            "observations.jsonl",
        )

    def test_legacy_evaluation(self):
        archive = self.legacy_run()

        original = {
            p.name: p.read_bytes()
            for p in archive.iterdir()
            if p.is_file()
        }

        destination, summary = evaluate_run(
            archive, SPEC
        )

        self.assertEqual(summary["observations"], 6)
        self.assertEqual(
            summary["npc_observations"], 3
        )

        metadata = json.loads(
            (destination / "manifest.json").read_text()
        )
        self.assertEqual(
            metadata["input_hashes"]["observation_source"],
            "events.jsonl",
        )

        for name, data in original.items():
            self.assertEqual(
                (archive / name).read_bytes(),
                data,
            )

    def test_incomplete_legacy_run(self):
        archive = self.legacy_run()
        path = archive / "events.jsonl"

        entries = [
            json.loads(line)
            for line in path.read_text().splitlines()
        ]
        entries = [
            e for e in entries
            if not (
                e["type"] == "response"
                and e["turn"] == 3
            )
        ]

        path.write_text(
            "".join(
                json.dumps(e) + "\n"
                for e in entries
            )
        )

        with self.assertRaises(ValueError):
            read_observations(archive)

        manifest_path = archive / "manifest.json"
        manifest = json.loads(
            manifest_path.read_text()
        )
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest)
        )

        observations = read_observations(archive)
        self.assertEqual(len(observations), 4)

    def test_legacy_gap_rejected(self):
        archive = self.legacy_run()
        path = archive / "events.jsonl"

        events = [
            json.loads(line)
            for line in path.read_text().splitlines()
        ]
        events = [
            e for e in events
            if not (
                e["type"] == "response"
                and e["turn"] == 2
            )
        ]

        path.write_text(
            "".join(
                json.dumps(e) + "\n"
                for e in events
            )
        )

        with self.assertRaises(ValueError):
            read_observations(archive)


    def test_early_selfplay_without_observations_file(self):
        archive = self.selfplay_run()

        # Simulate the original v0.4 archive format.
        (archive / "observations.jsonl").unlink()

        entries = read_observations(archive)

        self.assertEqual(len(entries), 6)
        self.assertEqual(
            [entry["role"] for entry in entries],
            ["player", "npc"] * 3,
        )
        self.assertEqual(
            source_path(archive).name, "events.jsonl"
        )

        destination, summary = evaluate_run(archive, SPEC)
        self.assertEqual(summary["observations"], 6)
        self.assertEqual(
            summary["npc_observations"], 3
        )
        self.assertTrue(
            (destination / "summary.json").is_file()
        )

    def test_early_selfplay_duplicate_response_rejected(self):
        archive = self.selfplay_run()
        (archive / "observations.jsonl").unlink()

        path = archive / "events.jsonl"
        events = [
            json.loads(line)
            for line in path.read_text().splitlines()
        ]
        duplicated = next(
            event for event in events
            if event.get("type") == "response"
            and event.get("turn") == 2
            and event.get("role") == "npc"
        )
        events.append(duplicated)

        path.write_text(
            "".join(json.dumps(event) + "\\n" for event in events),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            read_observations(archive)


if __name__ == "__main__":
    unittest.main()
