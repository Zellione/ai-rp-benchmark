import tempfile
import unittest
from pathlib import Path

from rpbench.embedding_cache import CachedEmbedder


CONFIG = {
    "id": "test-q8",
    "backend": {
        "model": "test-model",
        "protocol": "openai-embeddings",
    },
    "model_identity": {"sha256": "test-q8-hash"},
}


class EmbeddingCacheTests(unittest.TestCase):

    def test_cache_survives_reopening(self):
        calls = []

        def embedding(texts, config, *, prefix):
            calls.append((prefix, list(texts)))
            return [[float(len(t)), 1.0] for t in texts]

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "embeddings.sqlite"

            first = CachedEmbedder(db, CONFIG, embedding)
            a = first(
                ["hello", "world"],
                CONFIG,
                prefix="search_document",
            )

            second = CachedEmbedder(db, CONFIG, embedding)
            b = second(
                ["hello", "world"],
                CONFIG,
                prefix="search_document",
            )

            self.assertEqual(a, b)
            self.assertEqual(len(calls), 1)
            self.assertEqual(second.stats()["hits"], 2)
            self.assertEqual(second.stats()["misses"], 0)

    def test_query_and_document_are_separate(self):
        calls = []

        def embedding(texts, config, *, prefix):
            calls.append(prefix)
            return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            cache = CachedEmbedder(
                Path(tmp) / "cache.sqlite",
                CONFIG,
                embedding,
            )
            cache(["same"], CONFIG, prefix="search_document")
            cache(["same"], CONFIG, prefix="search_query")
            self.assertEqual(
                calls,
                ["search_document", "search_query"],
            )

    def test_model_identity_invalidates_cache(self):
        calls = []

        def embedding(texts, config, *, prefix):
            calls.append(1)
            return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cache.sqlite"

            first = CachedEmbedder(db, CONFIG, embedding)
            first(["text"], CONFIG, prefix="search_document")

            changed = {
                **CONFIG,
                "model_identity": {"sha256": "different-hash"},
            }
            second = CachedEmbedder(db, changed, embedding)
            second(["text"], changed, prefix="search_document")

            self.assertEqual(len(calls), 2)


    def test_repeated_connections_close_cleanly(self):
        import gc
        import warnings

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cache.sqlite"

            def embedding(texts, config, *, prefix):
                return [[1.0, 0.0] for _ in texts]

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error", ResourceWarning
                )

                cache = CachedEmbedder(
                    db, CONFIG, embedding
                )

                for index in range(30):
                    cache(
                        [f"document {index}"],
                        CONFIG,
                        prefix="search_document",
                    )

                gc.collect()

            reopened = CachedEmbedder(
                db, CONFIG, embedding
            )
            result = reopened(
                ["document 0"],
                CONFIG,
                prefix="search_document",
            )

            self.assertEqual(result, [[1.0, 0.0]])
            self.assertEqual(reopened.stats()["hits"], 1)

    def test_invalid_vector_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CachedEmbedder(
                Path(tmp) / "cache.sqlite",
                CONFIG,
                lambda texts, cfg, *, prefix: [
                    [0.0, 0.0] for _ in texts
                ],
            )
            with self.assertRaises(ValueError):
                cache(["text"], CONFIG, prefix="search_document")


if __name__ == "__main__":
    unittest.main()
