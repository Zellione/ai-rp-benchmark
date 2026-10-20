import unittest

from rpbench.retrieval_dense import (
    cosine,
    normalize,
    select_context,
)


CONFIG = {
    "backend": {
        "model": "mock",
        "base_url": "http://127.0.0.1:8082",
    }
}

SECRET = {
    "fact": "The vault key is hidden under the lantern."
}


class DenseRetrievalTests(unittest.TestCase):

    def test_cosine(self):
        self.assertAlmostEqual(
            cosine(normalize([3, 0]), normalize([1, 0])),
            1.0,
        )

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine([1, 0], [1, 0, 0])

    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError):
            normalize([0, 0])

    def test_semantic_selection(self):
        history = [
            {
                "turn": 1,
                "player": "Question about the vault.",
                "npc": "A key is displayed.",
            },
            {
                "turn": 2,
                "player": "What did Mara show Finn?",
                "npc": "She showed him the hiding place.",
            },
            {
                "turn": 3,
                "player": "Good morning.",
                "npc": "Good morning.",
            },
        ]

        def fake_embed(texts, config, *, prefix):
            if prefix == "search_query":
                return [[1, 0]]

            result = []
            for text in texts:
                if "hiding place" in text:
                    result.append([1, 0])
                else:
                    result.append([0, 1])
            return result

        selection = select_context(
            history,
            SECRET,
            window_turns=1,
            evidence_turns=1,
            config=CONFIG,
            embed_fn=fake_embed,
        )

        self.assertEqual(
            selection["retrieved_older_evidence"][0]["turn"],
            2,
        )

    def test_empty_older_history(self):
        result = select_context(
            [{
                "turn": 1,
                "player": "Hello",
                "npc": "Hi",
            }],
            SECRET,
            config=CONFIG,
            embed_fn=lambda *a, **kw: self.fail(
                "Should not embed"
            ),
        )

        self.assertTrue(
            result["coverage"]["history_complete"]
        )


if __name__ == "__main__":
    unittest.main()
