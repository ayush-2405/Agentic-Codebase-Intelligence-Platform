"""
ingestion/chunker.py — Splits source files into overlapping text chunks for
retrieval.

Each Chunk carries rich metadata (file path, type, functions/classes present,
summary excerpt) so that the retriever can surface relevant context even when
a query only matches metadata rather than raw code.

Strategy:
  1. If the file has identifiable top-level definitions (functions, classes),
     emit one chunk per definition plus its surrounding lines.
  2. If the file has no structural landmarks, fall back to sliding-window
     chunking on raw lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ingestion.loader import CodeFile
from ingestion.parser import ParsedModule
import config

logger = logging.getLogger(__name__)

# Lightweight offline token estimator (~4 chars/token for code)
_CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    """Estimate token count; accurate enough for chunking budget decisions."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens* tokens."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.8:
        return truncated[:last_nl]
    return truncated


@dataclass
class Chunk:
    """A retrievable unit of code."""

    chunk_id: str               # unique identifier: "<rel_path>::<idx>"
    file_path: str              # relative path within repo
    content: str                # the text that will be embedded & retrieved
    chunk_type: str             # "function" | "class" | "module" | "window"
    symbol_name: str            # function/class name or "" for window chunks
    summary_excerpt: str        # short summary blurb for this chunk
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.token_count = _count_tokens(self.content)


class CodeChunker:
    """
    Converts a CodeFile + ParsedModule into a list of Chunks.
    """

    def __init__(
        self,
        max_tokens: int = config.CHUNK_MAX_TOKENS,
        overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        code_file: CodeFile,
        parsed: ParsedModule | None = None,
        summary: str = "",
    ) -> list[Chunk]:
        """
        Return a list of Chunks for *code_file*.

        Args:
            code_file: Loaded source file.
            parsed:    AST metadata (None → fallback to window chunking).
            summary:   LLM-generated summary for this file.
        """
        lines = code_file.content.splitlines()
        chunks: list[Chunk] = []

        if parsed and not parsed.parse_error:
            chunks.extend(
                self._chunk_by_definitions(code_file, parsed, lines, summary)
            )

        # Always add a whole-file summary chunk so high-level queries hit it
        chunks.append(self._file_summary_chunk(code_file, summary))

        # Fallback: if no structural chunks were produced, use sliding windows
        if len(chunks) <= 1:
            chunks.extend(self._sliding_window_chunks(code_file, lines, summary))

        # Assign stable IDs
        for idx, chunk in enumerate(chunks):
            chunk.chunk_id = f"{code_file.rel_path}::{idx}"

        logger.debug("Produced %d chunks for %s", len(chunks), code_file.rel_path)
        return chunks

    # ── structural chunking ───────────────────────────────────────────────────

    def _chunk_by_definitions(
        self,
        code_file: CodeFile,
        parsed: ParsedModule,
        lines: list[str],
        summary: str,
    ) -> list[Chunk]:
        chunks = []
        definitions = []

        for func in parsed.functions:
            definitions.append(("function", func.name, func.lineno))
        for cls in parsed.classes:
            definitions.append(("class", cls.name, cls.lineno))

        definitions.sort(key=lambda x: x[2])

        for i, (def_type, name, lineno) in enumerate(definitions):
            # Determine end line: either next definition or EOF
            if i + 1 < len(definitions):
                end_line = definitions[i + 1][2] - 1
            else:
                end_line = len(lines)

            start_line = max(0, lineno - 1)  # 0-indexed
            snippet = "\n".join(lines[start_line:end_line])
            snippet = _truncate_to_tokens(snippet, self.max_tokens)

            # Build searchable content = header + code
            header = (
                f"File: {code_file.rel_path}\n"
                f"Type: {def_type}  Name: {name}\n"
                f"---\n"
            )
            content = header + snippet

            chunks.append(
                Chunk(
                    chunk_id="",  # assigned later
                    file_path=code_file.rel_path,
                    content=content,
                    chunk_type=def_type,
                    symbol_name=name,
                    summary_excerpt=summary[:200],
                    start_line=start_line + 1,
                    end_line=end_line,
                )
            )

        return chunks

    # ── file-level summary chunk ──────────────────────────────────────────────

    def _file_summary_chunk(self, code_file: CodeFile, summary: str) -> Chunk:
        content = (
            f"File: {code_file.rel_path}\n"
            f"Type: module summary\n"
            f"Lines: {code_file.lines}\n"
            f"---\n"
            f"{summary}"
        )
        return Chunk(
            chunk_id="",
            file_path=code_file.rel_path,
            content=_truncate_to_tokens(content, self.max_tokens),
            chunk_type="module",
            symbol_name="",
            summary_excerpt=summary[:200],
        )

    # ── sliding window fallback ───────────────────────────────────────────────

    def _sliding_window_chunks(
        self, code_file: CodeFile, lines: list[str], summary: str
    ) -> list[Chunk]:
        chunks = []
        window: list[str] = []
        window_tokens = 0
        start_line = 1

        for lineno, line in enumerate(lines, 1):
            line_tokens = _count_tokens(line + "\n")
            if window_tokens + line_tokens > self.max_tokens and window:
                content = (
                    f"File: {code_file.rel_path} (lines {start_line}-{lineno-1})\n---\n"
                    + "\n".join(window)
                )
                chunks.append(
                    Chunk(
                        chunk_id="",
                        file_path=code_file.rel_path,
                        content=content,
                        chunk_type="window",
                        symbol_name="",
                        summary_excerpt=summary[:200],
                        start_line=start_line,
                        end_line=lineno - 1,
                    )
                )
                # Overlap: keep the last N tokens worth of lines
                overlap_lines = self._take_last_tokens(window, self.overlap_tokens)
                window = overlap_lines
                window_tokens = sum(_count_tokens(l + "\n") for l in window)
                start_line = lineno - len(overlap_lines)

            window.append(line)
            window_tokens += line_tokens

        if window:
            content = (
                f"File: {code_file.rel_path} (lines {start_line}-{len(lines)})\n---\n"
                + "\n".join(window)
            )
            chunks.append(
                Chunk(
                    chunk_id="",
                    file_path=code_file.rel_path,
                    content=content,
                    chunk_type="window",
                    symbol_name="",
                    summary_excerpt=summary[:200],
                    start_line=start_line,
                    end_line=len(lines),
                )
            )

        return chunks

    @staticmethod
    def _take_last_tokens(lines: list[str], max_tokens: int) -> list[str]:
        result: list[str] = []
        tokens = 0
        for line in reversed(lines):
            t = _count_tokens(line + "\n")
            if tokens + t > max_tokens:
                break
            result.insert(0, line)
            tokens += t
        return result


def chunk_codebase(
    code_files: list[CodeFile],
    parsed_modules: dict[str, ParsedModule] | None = None,
    summaries: dict[str, str] | None = None,
) -> list[Chunk]:
    """
    Chunk an entire codebase at once.

    Args:
        code_files:     All loaded source files.
        parsed_modules: rel_path → ParsedModule mapping.
        summaries:      rel_path → summary string mapping.

    Returns:
        Flat list of all Chunk objects.
    """
    chunker = CodeChunker()
    all_chunks: list[Chunk] = []
    for cf in code_files:
        parsed = (parsed_modules or {}).get(cf.rel_path)
        summary = (summaries or {}).get(cf.rel_path, "")
        all_chunks.extend(chunker.chunk(cf, parsed, summary))
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks
