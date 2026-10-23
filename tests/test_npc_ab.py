import json
import tempfile
import unittest
from pathlib import Path

from rpbench.npc_ab import (
    build_specs,
    read_cases,
    run,
)


class NpcABTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        (self.root / "scenario.json").write_text(
            json.dumps({
                "system": "Portray Elara.",
                "character_cards": [],
            }),
            encoding="utf-8",
        )

        (self.root / "player.json").write_text(
            json.dumps({
                "name": "Rowan",
                "persona": "Careful",
                "main_goal": "Help",
                "subgoals": ["Talk"],
            }),
            encoding="utf-8",
        )

        self.cases = self.root / "cases.json"
        self.cases.write_text(
            json.dumps({
                "schema_version": 1,
                "cases": [{
                    "id": "one",
                    "language": "en",
                    "scenario": "scenario.json",
                    "context": "You are in the observatory.",
                    "player": '"Hello."',
                    "history": [],
                }],
            }),
            encoding="utf-8",
        )

        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps({
                "backend": {
                    "name": "mock",
                    "mode": "chat",
                    "model": "mock",
                    "base_url": "",
                },
                "sampling": {"max_tokens": 128},
            }),
            encoding="utf-8",
        )

    def test_reject_duplicate_case(self):
        data = json.loads(self.cases.read_text())
        data["cases"].append(dict(data["cases"][0]))
        self.cases.write_text(json.dumps(data))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_cases(self.cases)

    def test_messages_have_fixed_context(self):
        path, cases = read_cases(self.cases)
        config = json.loads(self.config.read_text())

        specs = build_specs(
            path,
            cases,
            self.config,
            config,
        )

        messages = specs[0]["payload"]["messages"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[-1]["content"],
            '"Hello."',
        )

    def test_resume_preserves_existing_response(self):
        output = self.root / "output"

        run(self.cases, self.config, output)

        original = (
            output / "cases" / "001.json"
        ).read_bytes()

        run(self.cases, self.config, output)

        self.assertEqual(
            original,
            (output / "cases" / "001.json").read_bytes(),
        )

        self.assertEqual(
            (
                output / "observations.jsonl"
            ).read_text().count("\n"),
            2,
        )

    def test_configuration_change_rejected(self):
        output = self.root / "output"

        run(self.cases, self.config, output)

        config = json.loads(self.config.read_text())
        config["sampling"]["max_tokens"] = 64

        self.config.write_text(json.dumps(config))

        with self.assertRaisesRegex(
            ValueError,
            "different configuration",
        ):
            run(self.cases, self.config, output)


if __name__ == "__main__":
    unittest.main()
