import unittest

from rpbench.attribution_diagnostics import (
    align_events,
    compare_pair,
    span_iou,
)


def event(start, end, speaker="mara-vey"):
    return {
        "start_offset": start,
        "end_offset": end,
        "kind": "speech",
        "speaker_id": speaker,
        "actor_id": None,
        "subject_id": None,
        "narrative_source": "direct_dialogue",
    }


class AttributionDiagnosticsTests(unittest.TestCase):

    def test_identical_span(self):
        a = event(10, 20)
        self.assertEqual(span_iou(a, a), 1.0)

    def test_partially_overlapping_span(self):
        a = event(10, 20)
        b = event(9, 21)
        self.assertAlmostEqual(
            span_iou(a, b), 10 / 12
        )

    def test_different_speaker_is_detected(self):
        a = event(10, 20, "mara-vey")
        b = event(10, 20, "finn-calder")
        result = compare_pair(a, b)

        self.assertFalse(
            result["role_matches"]["speaker_id"]
        )
        self.assertFalse(result["exact_match"])

    def test_no_double_assignment(self):
        gold = [event(10, 20)]
        predicted = [
            event(10, 20),
            event(10, 20, "finn-calder"),
        ]
        result = align_events(gold, predicted)

        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(
            result["unmatched_predicted"], 1
        )

    def test_nonoverlapping_events_remain_unmatched(self):
        result = align_events(
            [event(10, 20)],
            [event(30, 40)],
        )
        self.assertEqual(len(result["pairs"]), 0)
        self.assertEqual(result["unmatched_gold"], 1)


if __name__ == "__main__":
    unittest.main()
