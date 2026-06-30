"""
agents/retriever.py — Retriever Agent

Performs semantic search over the FAISS vector index and returns a
formatted context string ready for injection into the Reasoning agent's prompt.

Also tracks retrieval latency as a performance metric.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ingestion.indexer import SearchResult, VectorIndex, get_index
import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Output of the Retriever agent."""
    formatted_context: str       # ready-to-inject string for LLM prompts
    search_results: list[SearchResult]
    num_chunks_searched: int
    latency_ms: float
    query: str
    citations: list[dict]


class RetrieverAgent:
    """
    Wraps the VectorIndex with query-time logic:
      - Reformulates the query if needed (future: HyDE)
      - Deduplicates results by file
      - Formats context for downstream agents
    """

    def __init__(self, index: VectorIndex | None = None):
        self._index = index or get_index()

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        deduplicate_files: bool = False,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant code chunks for *query*.

        Args:
            query:             The user's question (or a sub-query).
            top_k:             How many chunks to return.
            deduplicate_files: If True, return at most one chunk per file.

        Returns:
            A RetrievalResult with formatted context and raw results.
        """
        t0 = time.perf_counter()
        results = self._index.search(query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        if deduplicate_files:
            seen_files: set[str] = set()
            deduped: list[SearchResult] = []
            for r in results:
                if r.chunk.file_path not in seen_files:
                    deduped.append(r)
                    seen_files.add(r.chunk.file_path)
            results = deduped

        formatted = self._format_context(results)
        logger.info(
            "Retrieved %d chunks in %.1fms for query: %.80s…",
            len(results), latency_ms, query
        )

        return RetrievalResult(
            formatted_context=formatted,
            search_results=results,
            num_chunks_searched=self._index.num_chunks,
            latency_ms=latency_ms,
            query=query,
            citations=self._collect_citations(results),
        )

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        if not results:
            return "No relevant code chunks found in the index."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            chunk = r.chunk
            header = (
                f"### [{i}] {chunk.file_path}"
                + (f" — {chunk.symbol_name}" if chunk.symbol_name else "")
                + f"  (score={r.score:.3f})"
            )
            citation = RetrieverAgent._citation_label(chunk.file_path, chunk.start_line)
            parts.append(f"{header}\nCitation: {citation}\n\n```\n{chunk.content}\n```")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _citation_label(file_path: str, start_line: int) -> str:
        line = start_line or 1
        return f"[source: {file_path}:{line}]"

    @classmethod
    def _collect_citations(cls, results: list[SearchResult]) -> list[dict]:
        citations: list[dict] = []
        seen: set[tuple[str, int, int]] = set()
        for result in results:
            chunk = result.chunk
            key = (chunk.file_path, chunk.start_line or 1, chunk.end_line or 0)
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line or 1,
                "end_line": chunk.end_line or chunk.start_line or 1,
                "label": cls._citation_label(chunk.file_path, chunk.start_line),
                "symbol_name": chunk.symbol_name,
                "chunk_type": chunk.chunk_type,
                "score": result.score,
            })
        return citations
