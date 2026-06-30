"""
ingestion/indexer.py — FAISS-backed vector index for semantic code retrieval.

The index stores:
  - L2-normalised embedding vectors (for cosine similarity via inner-product)
  - A parallel list of Chunk objects (the ground-truth payload)

Persistence: the index and chunk metadata are saved to disk so re-ingestion is
not required between sessions unless the codebase changes.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np

from ingestion.chunker import Chunk
from utils.embeddings import embed_batch, embed_text
import config

logger = logging.getLogger(__name__)

_INDEX_PATH = config.VECTOR_STORE_DIR / "faiss.index"
_CHUNKS_PATH = config.VECTOR_STORE_DIR / "chunks.pkl"
_VECTORS_PATH = config.VECTOR_STORE_DIR / "vectors.npy"


class SearchResult(NamedTuple):
    chunk: Chunk
    score: float   # cosine similarity [0, 1]
    rank: int


class VectorIndex:
    """
    Manages a FAISS flat inner-product index over code chunk embeddings.

    Using IP on L2-normalised vectors is equivalent to cosine similarity,
    which is more meaningful for semantic search than raw L2 distance.
    """

    def __init__(
        self,
        index_path: Path | None = None,
        chunks_path: Path | None = None,
        vectors_path: Path | None = None,
    ):
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._index_path = index_path or _INDEX_PATH
        self._chunks_path = chunks_path or _CHUNKS_PATH
        self._vectors_path = vectors_path or _VECTORS_PATH

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed all *chunks* and build the FAISS index from scratch.
        This overwrites any previously indexed data.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info("Embedding %d chunks …", len(chunks))
        texts = [c.content for c in chunks]
        vecs = embed_batch(texts)

        matrix = np.vstack(vecs).astype(np.float32)
        faiss.normalize_L2(matrix)

        self._set_index(matrix, chunks)
        logger.info("FAISS index built with %d vectors (dim=%d).", self._index.ntotal, config.EMBEDDING_DIM)

    def rebuild_incremental(self, chunks: list[Chunk], changed_files: set[str] | None = None) -> None:
        """
        Reuse stored vectors for unchanged files and only embed chunks for files
        whose content changed.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            self._index = None
            self._chunks = []
            self._vectors = None
            return

        changed_files = set(changed_files or set())
        if self._vectors is None or not len(self._chunks):
            self.build(chunks)
            return

        current_by_file = self._group_chunks_by_file(chunks)
        previous_by_file = self._group_chunk_vectors_by_file()
        vectors_by_file: dict[str, list[np.ndarray]] = {}
        files_to_embed: list[str] = []

        for file_path, file_chunks in current_by_file.items():
            if file_path in changed_files:
                files_to_embed.append(file_path)
                continue
            previous = previous_by_file.get(file_path)
            if previous is None:
                files_to_embed.append(file_path)
                continue
            prev_keys = [key for key, _ in previous]
            curr_keys = [self._chunk_cache_key(chunk) for chunk in file_chunks]
            if prev_keys != curr_keys:
                files_to_embed.append(file_path)
                continue
            vectors_by_file[file_path] = [vec for _, vec in previous]

        for file_path in files_to_embed:
            file_chunks = current_by_file[file_path]
            logger.info("Embedding changed file: %s (%d chunks)", file_path, len(file_chunks))
            vecs = embed_batch([chunk.content for chunk in file_chunks], batch_size=config.VECTOR_BATCH_SIZE)
            vectors_by_file[file_path] = [np.asarray(vec, dtype=np.float32) for vec in vecs]

        ordered_vectors: list[np.ndarray] = []
        file_offsets = {file_path: 0 for file_path in vectors_by_file}
        for chunk in chunks:
            idx = file_offsets[chunk.file_path]
            ordered_vectors.append(vectors_by_file[chunk.file_path][idx])
            file_offsets[chunk.file_path] = idx + 1

        matrix = np.vstack(ordered_vectors).astype(np.float32)
        faiss.normalize_L2(matrix)
        self._set_index(matrix, chunks)
        self._chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> list[SearchResult]:
        """
        Semantically search the index for chunks relevant to *query*.

        Args:
            query:     Free-text user query.
            top_k:     Maximum number of results to return.
            threshold: Minimum cosine similarity score to include.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Index is empty. Run build() first.")
            return []

        q_vec = embed_text(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)

        scores, indices = self._index.search(q_vec, min(top_k * 2, self._index.ntotal))

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            if float(score) < threshold:
                break
            results.append(SearchResult(chunk=self._chunks[idx], score=float(score), rank=rank))
            if len(results) >= top_k:
                break

        return results

    def get_chunk_by_file(self, rel_path: str) -> list[Chunk]:
        """Return all chunks belonging to a specific file."""
        return [c for c in self._chunks if c.file_path == rel_path]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index and chunk metadata to disk."""
        if self._index is None:
            logger.warning("Nothing to save — index is empty.")
            return

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))

        with open(self._chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

        logger.info("Index saved to %s (%d chunks).", self._index_path, len(self._chunks))

    def load(self) -> bool:
        """
        Load a previously saved index from disk.

        Returns:
            True if load succeeded, False if no saved index was found.
        """
        if not self._index_path.exists() or not self._chunks_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(self._index_path))
            with open(self._chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            if self._vectors_path.exists():
                self._vectors = np.load(self._vectors_path)
            else:
                self._vectors = None
            logger.info("Loaded index with %d chunks from %s.", len(self._chunks), self._index_path)
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    def _set_index(self, matrix: np.ndarray, chunks: list[Chunk]) -> None:
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        index.add(matrix)
        self._index = index
        self._chunks = chunks
        self._vectors = matrix

    def _group_chunk_vectors_by_file(self) -> dict[str, list[tuple[tuple, np.ndarray]]]:
        grouped: dict[str, list[tuple[tuple, np.ndarray]]] = {}
        if self._vectors is None:
            return grouped
        for chunk, vector in zip(self._chunks, self._vectors):
            grouped.setdefault(chunk.file_path, []).append((self._chunk_cache_key(chunk), vector))
        return grouped

    @staticmethod
    def _group_chunks_by_file(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.file_path, []).append(chunk)
        return grouped

    @staticmethod
    def _chunk_cache_key(chunk: Chunk) -> tuple:
        return (
            chunk.chunk_type,
            chunk.symbol_name,
            chunk.start_line,
            chunk.end_line,
            chunk.content,
        )


# Module-level singleton
_vector_index = VectorIndex()


def get_index() -> VectorIndex:
    """Return the module-level shared VectorIndex instance."""
    return _vector_index
