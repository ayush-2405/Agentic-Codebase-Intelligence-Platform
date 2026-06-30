"""
utils/embeddings.py — Embedding generation via OpenAI's embedding API.

Provides:
  - Single-text and batch embedding generation
  - Cosine similarity helper
  - Token-budget-aware batching
"""

from __future__ import annotations

import logging
import numpy as np

from utils.llm import get_client
import config

logger = logging.getLogger(__name__)


def embed_text(text: str) -> np.ndarray:
    """
    Generate a single embedding vector for *text*.

    Returns:
        A float32 numpy array of shape (EMBEDDING_DIM,).
    """
    client = get_client()
    text = text.replace("\n", " ").strip()
    if not text:
        return np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

    resp = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[text],
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec


def embed_batch(texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
    """
    Generate embeddings for a list of texts in batches.

    Args:
        texts:      List of strings to embed.
        batch_size: Number of texts per API call.

    Returns:
        List of float32 numpy arrays, one per input text.
    """
    client = get_client()
    all_vecs: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ").strip() or " " for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        # API guarantees order matches input
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        all_vecs.extend(vecs)
        logger.debug("Embedded batch %d/%d", i + len(batch), len(texts))

    return all_vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity ∈ [-1, 1] between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
