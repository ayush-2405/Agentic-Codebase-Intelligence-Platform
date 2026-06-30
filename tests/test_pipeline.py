import unittest
from pathlib import Path
from unittest.mock import patch
import shutil
from uuid import uuid4

import numpy as np

from agents.graph_agent import GraphAgent
from ingestion.indexer import VectorIndex
from ingestion.pipeline import ingest_repository


def _fake_embed_batch(texts, batch_size=64):
    vectors = []
    for idx, text in enumerate(texts, 1):
        vec = np.zeros(1536, dtype=np.float32)
        vec[idx % 1536] = float(len(text) or 1)
        vectors.append(vec)
    return vectors


class PipelineIncrementalTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_repository_reuses_unchanged_file_summaries(self):
        repo = Path(f"tests/_tmp_repo_{uuid4().hex}").resolve()
        store = Path(f"tests/_tmp_store_{uuid4().hex}").resolve()
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(store, ignore_errors=True)
        repo.mkdir(parents=True, exist_ok=True)
        store.mkdir(parents=True, exist_ok=True)
        try:
            (repo / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
            (repo / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")

            index = VectorIndex(
                index_path=store / "faiss.index",
                chunks_path=store / "chunks.pkl",
                vectors_path=store / "vectors.npy",
            )
            graph = GraphAgent(graph_path=store / "graph.pkl")
            manifest_path = store / "manifest.json"
            summary_calls: list[list[str]] = []

            async def fake_summaries(code_files, parsed_modules=None, existing_summaries=None, **_kwargs):
                summary_calls.append([cf.rel_path for cf in code_files])
                merged = dict(existing_summaries or {})
                for cf in code_files:
                    merged[cf.rel_path] = f"summary:{cf.rel_path}"
                return merged

            with patch("ingestion.indexer.embed_batch", side_effect=_fake_embed_batch), patch(
                "ingestion.pipeline.summarize_files_async",
                side_effect=fake_summaries,
            ):
                first = await ingest_repository(repo, index, graph, manifest_path)

            self.assertTrue({"a.py", "b.py"}.issubset(set(first.changed_files)))
            self.assertEqual(summary_calls[0], ["a.py", "b.py"])

            (repo / "b.py").write_text("def beta():\n    return 3\n", encoding="utf-8")
            index2 = VectorIndex(
                index_path=store / "faiss.index",
                chunks_path=store / "chunks.pkl",
                vectors_path=store / "vectors.npy",
            )
            graph2 = GraphAgent(graph_path=store / "graph.pkl")

            with patch("ingestion.indexer.embed_batch", side_effect=_fake_embed_batch), patch(
                "ingestion.pipeline.summarize_files_async",
                side_effect=fake_summaries,
            ):
                second = await ingest_repository(repo, index2, graph2, manifest_path)

            self.assertEqual(second.changed_files, ["b.py"])
            self.assertIn("a.py", second.reused_files)
            self.assertEqual(summary_calls[1], ["b.py"])
        finally:
            shutil.rmtree(repo, ignore_errors=True)
            shutil.rmtree(store, ignore_errors=True)
