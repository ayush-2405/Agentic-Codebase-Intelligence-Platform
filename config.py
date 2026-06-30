"""
config.py — Central configuration for the Agentic Codebase Intelligence Platform.
All tuneable parameters, paths, and environment variables live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "repo"
MEMORY_DIR = BASE_DIR / "memory"
VECTOR_STORE_DIR = MEMORY_DIR / "vector_store"
REPO_STORE_DIR = MEMORY_DIR / "repos"
REPO_REGISTRY_PATH = REPO_STORE_DIR / "registry.json"
GRAPH_DIR = BASE_DIR / "graph"
CHAT_HISTORY_PATH = MEMORY_DIR / "chat_history.json"
DEPENDENCY_GRAPH_PATH = GRAPH_DIR / "dependency_graph.pkl"

# Ensure directories exist
for d in [VECTOR_STORE_DIR, GRAPH_DIR, REPO_STORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Embedding model (used for semantic search)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = 1536  # dimensions for text-embedding-3-small

# ─── Ingestion ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: list[str] = [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h", ".hpp", ".rs", ".html"]
MAX_FILE_SIZE_KB: int = 500  # skip files larger than this
SUMMARIZER_MAX_CONCURRENCY: int = int(os.getenv("SUMMARIZER_MAX_CONCURRENCY", "8"))
VECTOR_BATCH_SIZE: int = int(os.getenv("VECTOR_BATCH_SIZE", "64"))

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 5
SIMILARITY_THRESHOLD: float = 0.25  # cosine similarity floor

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVALUATION_ENABLED: bool = os.getenv("EVALUATION_ENABLED", "true").lower() == "true"
MIN_ACCEPTABLE_SCORE: float = 0.6

# ─── Memory ───────────────────────────────────────────────────────────────────
MAX_HISTORY_ENTRIES: int = 500
MEMORY_SIMILARITY_TOP_K: int = 3
