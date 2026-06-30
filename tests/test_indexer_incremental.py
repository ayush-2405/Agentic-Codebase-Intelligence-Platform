import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

import numpy as np

from ingestion.chunker import Chunk
from ingestion.indexer import VectorIndex


def _vec_for(text: str) -> np.ndarray:
    seed = sum(ord(ch) for ch in text) % 97
    vec = np.zeros(1536, dtype=np.float32)
    vec[seed] = 1.0
    return vec


class IncrementalIndexerTests(unittest.TestCase):
    def test_rebuild_incremental_only_embeds_changed_files(self):
        base = Path("tests/_tmp_indexer").resolve()
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        try:
            index = VectorIndex(
                index_path=base / "faiss.index",
                chunks_path=base / "chunks.pkl",
                vectors_path=base / "vectors.npy",
            )

            initial_chunks = [
                Chunk("a::0", "a.py", "alpha", "module", "", "a", 1, 1),
                Chunk("b::0", "b.py", "beta", "module", "", "b", 1, 1),
            ]

            embed_calls: list[list[str]] = []

            def fake_embed_batch(texts, batch_size=64):
                embed_calls.append(list(texts))
                return [_vec_for(text) for text in texts]

            with patch("ingestion.indexer.embed_batch", side_effect=fake_embed_batch):
                index.build(initial_chunks)
                index.save()

            reloaded = VectorIndex(
                index_path=base / "faiss.index",
                chunks_path=base / "chunks.pkl",
                vectors_path=base / "vectors.npy",
            )
            self.assertTrue(reloaded.load())

            updated_chunks = [
                Chunk("a::0", "a.py", "alpha", "module", "", "a", 1, 1),
                Chunk("b::0", "b.py", "beta changed", "module", "", "b", 1, 1),
            ]

            with patch("ingestion.indexer.embed_batch", side_effect=fake_embed_batch):
                reloaded.rebuild_incremental(updated_chunks, {"b.py"})

            self.assertEqual(embed_calls[0], ["alpha", "beta"])
            self.assertEqual(embed_calls[1], ["beta changed"])
            self.assertEqual(reloaded.num_chunks, 2)
        finally:
            shutil.rmtree(base, ignore_errors=True)
