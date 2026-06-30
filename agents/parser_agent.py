"""
agents/parser_agent.py — Parser Agent

On-demand structural metadata extraction for files identified by the retriever
or requested explicitly by the orchestrator.

Unlike the batch ingestion parser, this agent operates at query time and can:
  - Parse specific files mentioned in the query
  - Return structured metadata for the reasoning agent
  - Detect when a query references a specific function/class
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ingestion.loader import CodeFile
from ingestion.parser import ParsedModule, parse_source_file
import config

logger = logging.getLogger(__name__)


class ParserAgent:
    """
    Provides structured metadata for files in the repo.
    Uses the pre-parsed module cache (from ingestion) when available,
    falls back to live parsing otherwise.
    """

    def __init__(self, parsed_cache: dict[str, ParsedModule] | None = None):
        """
        Args:
            parsed_cache: Pre-computed rel_path → ParsedModule mapping from
                          ingestion. Avoids re-parsing during queries.
        """
        self._cache: dict[str, ParsedModule] = parsed_cache or {}

    def get_metadata(self, rel_paths: list[str]) -> str:
        """
        Return a formatted string of structural metadata for the given file paths.

        Args:
            rel_paths: List of repo-relative file paths to analyse.

        Returns:
            A markdown-formatted metadata string for injection into LLM prompts.
        """
        parts: list[str] = []
        for rel_path in rel_paths:
            parsed = self._get_or_parse(rel_path)
            if parsed:
                parts.append(self._format_module(parsed))
            else:
                parts.append(f"⚠️ Could not parse: {rel_path}")

        return "\n\n".join(parts) if parts else "No parser metadata available."

    def find_symbol(self, symbol_name: str) -> list[dict]:
        """
        Search all cached modules for a function or class matching *symbol_name*.

        Returns:
            List of dicts with 'file', 'type', 'lineno', 'args'.
        """
        results = []
        for rel_path, parsed in self._cache.items():
            for func in parsed.functions:
                if func.name == symbol_name:
                    results.append({
                        "file": rel_path,
                        "type": "function",
                        "lineno": func.lineno,
                        "args": func.args,
                        "docstring": func.docstring[:100],
                    })
            for cls in parsed.classes:
                if cls.name == symbol_name:
                    results.append({
                        "file": rel_path,
                        "type": "class",
                        "lineno": cls.lineno,
                        "methods": [m.name for m in cls.methods],
                    })
                for method in cls.methods:
                    if method.name == symbol_name:
                        results.append({
                            "file": rel_path,
                            "type": "method",
                            "class": cls.name,
                            "lineno": method.lineno,
                            "args": method.args,
                        })
        return results

    def update_cache(self, rel_path: str, parsed: ParsedModule) -> None:
        """Add or replace a module in the cache."""
        self._cache[rel_path] = parsed

    # ── private ──────────────────────────────────────────────────────────────

    def _get_or_parse(self, rel_path: str) -> ParsedModule | None:
        if rel_path in self._cache:
            return self._cache[rel_path]

        # Try to parse from disk
        full_path = config.DATA_DIR / rel_path
        if not full_path.exists():
            logger.warning("File not found: %s", full_path)
            return None

        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_source_file(source, rel_path)
            self._cache[rel_path] = parsed
            return parsed
        except Exception as exc:
            logger.error("Live parse failed for %s: %s", rel_path, exc)
            return None

    @staticmethod
    def _format_module(parsed: ParsedModule) -> str:
        lines = [f"#### `{parsed.file_path}`"]
        if parsed.parse_error:
            lines.append(f"⚠️ Parse error: {parsed.parse_error}")
            return "\n".join(lines)

        if parsed.imports:
            imp_names = ", ".join(i.module for i in parsed.imports[:8])
            lines.append(f"**Imports:** {imp_names}")

        if parsed.classes:
            for cls in parsed.classes:
                method_names = ", ".join(m.name for m in cls.methods)
                lines.append(f"**Class** `{cls.name}` → methods: {method_names or '(none)'}")

        if parsed.functions:
            for fn in parsed.functions:
                arg_str = ", ".join(fn.args)
                doc = f" — {fn.docstring[:80]}" if fn.docstring else ""
                lines.append(f"**Function** `{fn.name}({arg_str})`{doc}")

        return "\n".join(lines)
