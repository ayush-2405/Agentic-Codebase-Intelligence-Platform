"""
agents/tool_agent.py — Tool Agent

Exposes deterministic filesystem tools that other agents (and the orchestrator)
can call directly:

  read_file(path)           — Return raw source of a file
  search_repo(keyword)      — Find files containing a keyword
  list_files(extension)     — List all files with a given extension
  get_file_stats(path)      — Return metadata about a file

These tools produce exact results, unlike the LLM-backed agents.  The
Reasoning agent uses their output as grounding context.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class ToolAgent:
    """Exposes safe, read-only filesystem tools for the codebase repo."""

    def __init__(self, root_dir: Path | None = None):
        self._root = root_dir or config.DATA_DIR

    # ── Public tools ─────────────────────────────────────────────────────────

    def read_file(self, rel_path: str) -> str:
        """
        Return the raw content of *rel_path* within the repo.

        Args:
            rel_path: Path relative to the repo root.

        Returns:
            File content string, or an error message if unreadable.
        """
        full_path = self._root / rel_path
        if not full_path.exists():
            return f"[ToolAgent] File not found: {rel_path}"
        if not full_path.is_file():
            return f"[ToolAgent] Not a file: {rel_path}"

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            logger.debug("read_file: %s (%d bytes)", rel_path, len(content))
            return content
        except OSError as exc:
            return f"[ToolAgent] Error reading {rel_path}: {exc}"

    def search_repo(
        self,
        keyword: str,
        extensions: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 20,
    ) -> list[dict]:
        """
        Find all files containing *keyword* (regex supported).

        Args:
            keyword:        Search term (plain string or regex).
            extensions:     File extensions to restrict search to.
            case_sensitive: Default False.
            max_results:    Cap on returned results.

        Returns:
            List of dicts: {file, line_no, line_content}
        """
        exts = set(extensions or config.SUPPORTED_EXTENSIONS)
        flags = 0 if case_sensitive else re.IGNORECASE
        if regex:
            try:
                pattern = re.compile(keyword, flags)
            except re.error:
                pattern = re.compile(re.escape(keyword), flags)
        else:
            pattern = re.compile(re.escape(keyword), flags)

        results: list[dict] = []
        for path in self._root.rglob("*"):
            if not path.is_file() or path.suffix not in exts:
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            try:
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if pattern.search(line):
                        results.append({
                            "file": str(path.relative_to(self._root)),
                            "line_no": lineno,
                            "line_content": line.strip()[:200],
                        })
                        if len(results) >= max_results:
                            return results
            except OSError:
                continue
        return results

    def list_files(self, extension: str | None = None) -> list[str]:
        """
        List all source files in the repo.

        Args:
            extension: If given (e.g. ".py"), only return files with this ext.

        Returns:
            Sorted list of relative file paths.
        """
        exts = {extension} if extension else set(config.SUPPORTED_EXTENSIONS)
        paths = []
        for path in self._root.rglob("*"):
            if path.is_file() and path.suffix in exts:
                if not any(part.startswith(".") for part in path.parts):
                    paths.append(str(path.relative_to(self._root)))
        return sorted(paths)

    def get_file_stats(self, rel_path: str) -> dict:
        """Return metadata about a file (size, lines, extension)."""
        full_path = self._root / rel_path
        if not full_path.exists():
            return {"error": f"File not found: {rel_path}"}
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return {
                "file": rel_path,
                "size_bytes": full_path.stat().st_size,
                "lines": content.count("\n") + 1,
                "extension": full_path.suffix,
            }
        except OSError as exc:
            return {"error": str(exc)}

    def format_search_results(self, results: list[dict]) -> str:
        """Format search_repo results for LLM prompt injection."""
        if not results:
            return "No matches found."
        lines = []
        for r in results:
            lines.append(f"  {r['file']}:{r['line_no']}  →  {r['line_content']}")
        return "\n".join(lines)
