"""
ingestion/loader.py — Recursively discover and load source files from the repo.

Design goals:
  - Pure filesystem operation; no LLM calls here.
  - Returns a list of CodeFile dataclass instances.
  - Skips binary files, oversized files, and hidden directories.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import config

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a single source file loaded from disk."""

    path: Path                  # absolute path
    rel_path: str               # path relative to DATA_DIR
    extension: str              # e.g. ".py"
    content: str                # raw source text
    size_bytes: int = 0
    lines: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.size_bytes = len(self.content.encode("utf-8"))
        self.lines = self.content.count("\n") + 1
        self.metadata.setdefault("sha256", hashlib.sha256(self.content.encode("utf-8")).hexdigest())


class RepoLoader:
    """
    Walks *root_dir* and loads all source files whose extension is in
    *supported_extensions*.
    """

    def __init__(
        self,
        root_dir: Path | None = None,
        supported_extensions: list[str] | None = None,
        max_file_size_kb: int | None = None,
    ):
        self.root_dir = root_dir or config.DATA_DIR
        self.extensions = set(supported_extensions or config.SUPPORTED_EXTENSIONS)
        self.max_size_bytes = (max_file_size_kb or config.MAX_FILE_SIZE_KB) * 1024

    def load(self) -> list[CodeFile]:
        """
        Walk the repo directory and return a list of CodeFile objects.

        Returns:
            List of loaded CodeFile instances (skipped files are logged).
        """
        if not self.root_dir.exists():
            logger.warning("Repo directory does not exist: %s", self.root_dir)
            return []

        files: list[CodeFile] = []
        for path in self._walk():
            cf = self._load_file(path)
            if cf is not None:
                files.append(cf)

        logger.info("Loaded %d source files from %s", len(files), self.root_dir)
        return files

    # ── private ──────────────────────────────────────────────────────────────

    def _walk(self):
        """Yield all files with a supported extension, skipping hidden dirs."""
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix in self.extensions:
                # Skip anything inside hidden directories (e.g., .git, .venv)
                if any(part.startswith(".") for part in path.parts):
                    continue
                yield path

    def _load_file(self, path: Path) -> CodeFile | None:
        """Read *path* and return a CodeFile, or None if it should be skipped."""
        if path.stat().st_size > self.max_size_bytes:
            logger.debug("Skipping large file: %s", path)
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return None

        rel_path = str(path.relative_to(self.root_dir))
        return CodeFile(
            path=path,
            rel_path=rel_path,
            extension=path.suffix,
            content=content,
        )
