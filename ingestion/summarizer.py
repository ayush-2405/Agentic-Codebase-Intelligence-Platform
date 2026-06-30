"""
ingestion/summarizer.py — LLM-based file summarisation.

Each CodeFile is summarised into a short natural-language description covering:
  - Purpose of the file
  - Key classes / functions
  - External dependencies
  - Role in the larger system (best-effort)

Summaries are stored alongside chunks in the vector index so that semantic
search can match high-level intent queries ("where is the auth logic?") to
the right files even when the exact code isn't retrieved.
"""

from __future__ import annotations

import asyncio
import logging

from ingestion.loader import CodeFile
from ingestion.parser import ParsedModule
from utils import llm, prompts
import config

logger = logging.getLogger(__name__)


class FileSummarizer:
    """Generates LLM summaries for source files."""

    def summarize(self, code_file: CodeFile, parsed: ParsedModule | None = None) -> str:
        """
        Return a concise natural-language summary for *code_file*.

        Args:
            code_file: The loaded source file.
            parsed:    Optional pre-parsed AST metadata; included in the prompt
                       if provided to give the LLM accurate structure info.

        Returns:
            A string summary (≤ 200 words).
        """
        # Build a richer prompt when AST metadata is available
        code_snippet = code_file.content[:3000]  # cap to avoid huge prompts
        extra = ""
        if parsed and not parsed.parse_error:
            functions = [f.name for f in parsed.functions]
            classes = [c.name for c in parsed.classes]
            imports = [i.module for i in parsed.imports[:10]]
            extra = (
                f"\n\nAST-extracted info:"
                f"\n  Classes: {classes}"
                f"\n  Functions: {functions}"
                f"\n  Imports: {imports}"
            )

        user_msg = prompts.summarizer_user(code_file.rel_path, code_snippet) + extra

        try:
            summary = llm.system_user(
                system=prompts.SUMMARIZER_SYSTEM,
                user=user_msg,
                max_tokens=300,
                temperature=0.1,
            )
            return summary.strip()
        except Exception as exc:
            logger.warning("Summarisation failed for %s: %s", code_file.rel_path, exc)
            # Fallback: return a minimal structural summary
            return self._fallback_summary(code_file, parsed)

    @staticmethod
    def _fallback_summary(code_file: CodeFile, parsed: ParsedModule | None) -> str:
        lines = [f"File: {code_file.rel_path}", f"Lines: {code_file.lines}"]
        if parsed:
            if parsed.functions:
                lines.append("Functions: " + ", ".join(f.name for f in parsed.functions[:5]))
            if parsed.classes:
                lines.append("Classes: " + ", ".join(c.name for c in parsed.classes[:5]))
            if parsed.imports:
                lines.append("Imports: " + ", ".join(i.module for i in parsed.imports[:5]))
        return " | ".join(lines)


def summarize_files(
    code_files: list[CodeFile],
    parsed_modules: dict[str, ParsedModule] | None = None,
) -> dict[str, str]:
    """
    Summarise a list of files and return {rel_path: summary}.

    Args:
        code_files:     Files to summarise.
        parsed_modules: Optional dict mapping rel_path → ParsedModule.

    Returns:
        Mapping from relative file path to summary string.
    """
    summarizer = FileSummarizer()
    summaries: dict[str, str] = {}
    total = len(code_files)
    for idx, cf in enumerate(code_files, 1):
        parsed = (parsed_modules or {}).get(cf.rel_path)
        logger.info("Summarising [%d/%d]: %s", idx, total, cf.rel_path)
        summaries[cf.rel_path] = summarizer.summarize(cf, parsed)
    return summaries


async def summarize_files_async(
    code_files: list[CodeFile],
    parsed_modules: dict[str, ParsedModule] | None = None,
    existing_summaries: dict[str, str] | None = None,
    max_concurrency: int | None = None,
    on_progress=None,
) -> dict[str, str]:
    """
    Summarise files concurrently while preserving existing summaries when supplied.
    """
    if not code_files:
        return dict(existing_summaries or {})

    summarizer = FileSummarizer()
    summaries: dict[str, str] = dict(existing_summaries or {})
    total = len(code_files)
    completed = 0
    limiter = asyncio.Semaphore(max(1, max_concurrency or config.SUMMARIZER_MAX_CONCURRENCY))

    async def _summarize(code_file: CodeFile) -> tuple[str, str]:
        async with limiter:
            parsed = (parsed_modules or {}).get(code_file.rel_path)
            logger.info("Summarising async: %s", code_file.rel_path)
            summary = await asyncio.to_thread(summarizer.summarize, code_file, parsed)
            return code_file.rel_path, summary

    tasks = [asyncio.create_task(_summarize(cf)) for cf in code_files]
    for task in asyncio.as_completed(tasks):
        rel_path, summary = await task
        summaries[rel_path] = summary
        completed += 1
        if on_progress is not None:
            maybe_awaitable = on_progress(completed, total, rel_path)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
    return summaries
