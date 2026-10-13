import json
import tempfile
import unittest
from pathlib import Path
from rpbench.core import Backend, raw_prompt, run

ROOT = Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    def test_request_mode(self):
        b = Backend("llamacpp", "http://localhost:8080", "foo", "chat")
        url, payload = b.prepare([{"role": "user", "content": "Hi"}], {"temperature": .8}, {})
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(payload["temperature"], .8)

    def test_kobold_native(self):
        b = Backend("koboldcpp-native", "http://localhost:5001", "foo", "completion")
        url, payload = b.prepare([{"role": "user", "content": "Hi"}], {"max_tokens": 42}, {})
        self.assertEqual(url, "http://localhost:5001/api/v1/generate")
        self.assertEqual(payload["max_length"], 42)
        self.assertIn("Assistant:", payload["prompt"])

    def test_run_logs_all_turns(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = run(str(ROOT / "scenarios/suspicious-archivist.json"),
                          str(ROOT / "configs/mock.json"), temp)
            manifest = json.loads((archive / "manifest.json").read_text())
            lines = (archive / "events.jsonl").read_text().splitlines()
            transcript = (archive / "conversation.md").read_text()
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(lines), 6)
            self.assertEqual(transcript.count("## Turn"), 3)

if __name__ == "__main__":
    unittest.main()
