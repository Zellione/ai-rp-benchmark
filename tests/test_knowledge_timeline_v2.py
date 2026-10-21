import tempfile
import unittest
from pathlib import Path

from rpbench.core import dump
from rpbench.knowledge_timeline import make_claims
from rpbench.knowledge_timeline_v2 import (
    build_timeline,
    make_transfers,
    merge_events,
    sha,
)
from rpbench.observations import source_path


OBS = [
    {"turn": 1, "role": "player", "text": "Hello."},
    {
        "turn": 1,
        "role": "npc",
        "text": "Mara tells Finn the vault code is 7319.",
    },
    {"turn": 2, "role": "player", "text": "And then?"},
    {
        "turn": 2,
        "role": "npc",
        "text": "Finn says Mara told him the vault code.",
    },
]

IDS = {"mara-vey", "finn-calder"}


def transfer(turn=1, kind="explicit_transfer", quote=None):
    if quote is None:
        quote = (
            "Mara tells Finn the vault code is 7319."
            if turn == 1 else
            "Mara told him the vault code"
        )

    return {
        "turn": turn,
        "role": "npc",
        "kind": kind,
        "source_character_id": "mara-vey",
        "recipient_character_id": "finn-calder",
        "fact": "The vault code is 7319.",
        "recipient_prior_knowledge": "unknown",
        "quote": quote,
        "start_offset": None,
        "end_offset": None,
        "explanation": "Candidate transfer.",
        "status": "candidate",
        "verified": False,
        "acquisition_confirmed": False,
    }


def write_jsonl(path, entries):
    import json
    path.write_text(
        "".join(
            json.dumps(entry) + "\n"
            for entry in entries
        ),
        encoding="utf-8",
    )


class KnowledgeTimelineV2Tests(unittest.TestCase):

    def test_transfer_candidate(self):
        result = make_transfers(
            OBS, [transfer()], IDS
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["turn"], 1)
        self.assertEqual(
            result[0]["recipient_character_id"],
            "finn-calder",
        )
        self.assertFalse(result[0]["transfer_confirmed"])
        self.assertFalse(
            result[0]["knowledge_acquisition_confirmed"]
        )

    def test_reported_transfer_not_confirmed(self):
        result = make_transfers(
            OBS, [transfer(2, "reported_transfer")], IDS
        )
        self.assertEqual(
            result[0]["kind"], "reported_transfer"
        )
        self.assertFalse(result[0]["verified"])

    def test_fabricated_quote_rejected(self):
        event = transfer()
        event["quote"] = "The dragon owns the library."
        with self.assertRaisesRegex(ValueError, "not found"):
            make_transfers(OBS, [event], IDS)

    def test_unknown_recipient_rejected(self):
        event = transfer()
        event["recipient_character_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "recipient"):
            make_transfers(OBS, [event], IDS)

    def test_claimed_confirmation_rejected(self):
        event = transfer()
        event["acquisition_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "confirmed"):
            make_transfers(OBS, [event], IDS)

    def test_ambiguous_quote_rejected(self):
        repeated = [
            {"turn": 1, "role": "npc", "text": "secret secret"}
        ]
        event = transfer()
        event["quote"] = "secret"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            make_transfers(repeated, [event], IDS)

    def test_merge_sort_and_provenance(self):
        claims = make_claims(
            OBS,
            [{
                "kind": "knowledge_claim",
                "turn": 2,
                "character_id": "finn-calder",
                "quote": "Finn says",
                "start_offset": None,
                "end_offset": None,
            }],
            IDS,
        )
        transfers = make_transfers(
            OBS, [transfer()], IDS
        )
        events = merge_events(claims, transfers)
        self.assertEqual(
            [e["turn"] for e in events],
            [1, 2],
        )
        self.assertEqual(
            [e["timeline_id"] for e in events],
            [1, 2],
        )
        self.assertEqual(
            events[0]["evidence_source"],
            "knowledge_transfers",
        )
        self.assertEqual(
            events[1]["evidence_source"],
            "character_attribution",
        )

    def test_full_build_without_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            attr = root / "evaluations" / "attr"
            trans = root / "evaluations" / "trans"
            attr.mkdir(parents=True)
            trans.mkdir(parents=True)

            dump(root / "manifest.json", {
                "type": "goal_driven_selfplay",
                "status": "complete",
                "completed_turns": 2,
            })
            write_jsonl(
                root / "observations.jsonl",
                [
                    {
                        "type": "dialogue_observation",
                        **entry,
                    }
                    for entry in OBS
                ],
            )
            catalog = [
                {"id": "mara-vey", "name": "Mara"},
                {"id": "finn-calder", "name": "Finn"},
            ]
            source_hash = sha(source_path(root))
            run_hash = sha(root / "manifest.json")

            dump(attr / "manifest.json", {
                "identity": {
                    "version": "character-attribution-v2",
                    "source_sha256": source_hash,
                    "run_manifest_sha256": run_hash,
                },
                "catalog": catalog,
            })
            dump(trans / "manifest.json", {
                "identity": {
                    "version": "knowledge-transfers-v2",
                    "source_hash": source_hash,
                    "manifest_hash": run_hash,
                },
                "catalog": catalog,
            })
            dump(attr / "summary.json", {
                "status": "complete",
            })
            dump(trans / "summary.json", {
                "status": "complete",
            })
            write_jsonl(attr / "findings.jsonl", [])
            write_jsonl(
                trans / "findings.jsonl",
                [
                    transfer(),
                    transfer(2, "reported_transfer"),
                ],
            )

            target, summary = build_timeline(
                root, attr, trans
            )
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["explicit_candidates"], 1)
            self.assertEqual(summary["reported_candidates"], 1)
            self.assertEqual(summary["confirmed_transfers"], 0)
            self.assertTrue(
                (target / "events.jsonl").is_file()
            )
            self.assertFalse(
                (root / "state.json").exists()
            )

            with self.assertRaises(FileExistsError):
                build_timeline(root, attr, trans)

    def test_partial_transfer_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "attribution").mkdir()
            (root / "transfers").mkdir()
            with self.assertRaises(ValueError):
                build_timeline(
                    root,
                    root / "attribution",
                    root / "transfers",
                )


if __name__ == "__main__":
    unittest.main()
