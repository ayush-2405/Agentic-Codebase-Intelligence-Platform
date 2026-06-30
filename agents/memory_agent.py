"""
agents/memory_agent.py — Memory Agent

Maintains persistent query history and enables retrieval of similar past
interactions to enrich future answers (contextual memory).

Storage: JSON file on disk (simple, portable, inspectable).

Future extensions:
  - Upgrade to a vector-backed memory for semantic past-query retrieval
  - Add session-level short-term memory ring buffer
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from utils.embeddings import cosine_similarity, embed_text
import config

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single stored interaction."""
    query: str
    answer: str
    timestamp: float = field(default_factory=time.time)
    evaluation_score: float = 0.0
    agent_plan: list[str] = field(default_factory=list)
    retrieved_files: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    embedding: list[float] = field(default_factory=list)  # serialised for JSON

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class MemoryAgent:
    """
    Stores, retrieves, and summarises past query-answer pairs.

    Memory is persisted to *chat_history.json* between sessions.
    Semantic similarity search over past queries is supported when embeddings
    are available.
    """

    def __init__(self, history_path: Path | None = None):
        self._path = history_path or config.CHAT_HISTORY_PATH
        self._entries: list[MemoryEntry] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, entry: MemoryEntry) -> None:
        """
        Persist a new interaction to memory.

        Generates an embedding for the query so semantic retrieval works later.
        Trims history to MAX_HISTORY_ENTRIES.
        """
        # Embed the query for future similarity search
        if not entry.embedding:
            try:
                vec = embed_text(entry.query)
                entry.embedding = vec.tolist()
            except Exception as exc:
                logger.warning("Could not embed memory entry: %s", exc)

        self._entries.append(entry)

        # Trim oldest entries
        if len(self._entries) > config.MAX_HISTORY_ENTRIES:
            self._entries = self._entries[-config.MAX_HISTORY_ENTRIES :]

        self._save()
        logger.debug("Stored memory entry #%d", len(self._entries))

    def retrieve_similar(self, query: str, top_k: int = config.MEMORY_SIMILARITY_TOP_K) -> list[MemoryEntry]:
        """
        Return the *top_k* past entries most similar to *query*.

        Falls back to the most recent entries if embeddings are unavailable.
        """
        if not self._entries:
            return []

        # Attempt semantic similarity
        try:
            q_vec = embed_text(query)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in self._entries:
                if entry.embedding:
                    import numpy as np
                    e_vec = np.array(entry.embedding, dtype="float32")
                    sim = cosine_similarity(q_vec, e_vec)
                    scored.append((sim, entry))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                return [e for _, e in scored[:top_k]]
        except Exception as exc:
            logger.warning("Semantic memory retrieval failed: %s", exc)

        # Fallback: most recent
        return self._entries[-top_k:]

    def format_past_context(self, entries: list[MemoryEntry]) -> str:
        """Format past entries for injection into the Reasoning agent prompt."""
        if not entries:
            return ""
        parts = []
        for e in entries:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
            parts.append(f"**[{ts}] Q:** {e.query}\n**A:** {e.answer[:400]}…")
        return "\n\n".join(parts)

    def recent_summary(self, n: int = 5) -> str:
        """Return a compact summary of the last *n* queries."""
        recent = self._entries[-n:]
        if not recent:
            return ""
        lines = [f"- {e.query}" for e in recent]
        return "Recent queries:\n" + "\n".join(lines)

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        """Wipe all history (non-reversible)."""
        self._entries = []
        self._save()
        logger.info("Memory cleared.")

    @property
    def count(self) -> int:
        return len(self._entries)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except OSError as exc:
            logger.error("Failed to save memory: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            self._entries = [MemoryEntry.from_dict(d) for d in raw]
            logger.info("Loaded %d memory entries from %s.", len(self._entries), self._path)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Could not load memory: %s. Starting fresh.", exc)
            self._entries = []
