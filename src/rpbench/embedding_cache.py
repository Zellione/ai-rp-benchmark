"""Persistent, content-addressed cache for text embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path


SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


class CachedEmbedder:
    def __init__(self, path, config, embed_fn):
        self.path = Path(path)
        self.config = config
        self.embed_fn = embed_fn
        self.hits = 0
        self.misses = 0

        backend = config["backend"]
        self.namespace = digest({
            "schema": SCHEMA_VERSION,
            "model": backend["model"],
            "model_identity": config.get(
                "model_identity", config["id"]
            ),
            "protocol": backend["protocol"],
        })

        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with closing(self.connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL
                )
            """)

    def connect(self):
        db = sqlite3.connect(
            self.path, timeout=30
        )
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def key(self, prefix, text):
        return digest({
            "namespace": self.namespace,
            "prefix": prefix,
            "text": text,
        })

    @staticmethod
    def validate_vector(vector):
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError("Invalid cached embedding")
        if sum(x * x for x in vector) <= 0:
            raise ValueError("Zero embedding")
        return vector

    def __call__(self, texts, config, *, prefix):
        if prefix not in {
            "search_document", "search_query"
        }:
            raise ValueError("Unknown embedding prefix")
        if not texts:
            return []
        if any(not isinstance(t, str) for t in texts):
            raise ValueError("Embedding texts must be strings")

        keys = [self.key(prefix, t) for t in texts]
        found = {}
        missing = {}

        with closing(self.connect()) as db, db:
            for index, (key, text) in enumerate(
                zip(keys, texts)
            ):
                if key in found:
                    continue

                row = db.execute(
                    "SELECT dimensions, vector_json "
                    "FROM embeddings WHERE key = ? "
                    "AND namespace = ?",
                    (key, self.namespace),
                ).fetchone()

                if row is not None:
                    vector = self.validate_vector(
                        json.loads(row[1])
                    )
                    if len(vector) != row[0]:
                        raise ValueError(
                            "Cached embedding dimension mismatch"
                        )
                    found[key] = vector
                else:
                    missing[key] = text

            if missing:
                unique_keys = list(missing)
                generated = self.embed_fn(
                    [missing[k] for k in unique_keys],
                    config,
                    prefix=prefix,
                )

                if len(generated) != len(unique_keys):
                    raise ValueError(
                        "Embedding result count mismatch"
                    )

                for key, vector in zip(
                    unique_keys, generated
                ):
                    vector = self.validate_vector(vector)
                    found[key] = vector

                db.executemany(
                    "INSERT OR IGNORE INTO embeddings "
                    "(key, namespace, dimensions, vector_json) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (
                            key,
                            self.namespace,
                            len(found[key]),
                            json.dumps(found[key]),
                        )
                        for key in unique_keys
                    ],
                )

        self.hits += sum(
            key not in missing for key in keys
        )
        self.misses += sum(
            key in missing for key in keys
        )

        result = [found[key] for key in keys]
        if len({len(v) for v in result}) != 1:
            raise ValueError(
                "Mixed embedding dimensions"
            )
        return result

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "namespace": self.namespace,
            "cache_path": str(self.path),
        }
