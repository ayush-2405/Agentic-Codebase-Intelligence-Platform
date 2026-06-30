"""
agents/reasoning_agent.py — Reasoning Agent

The synthesis layer of the pipeline. Receives structured context from all
upstream agents and produces the final natural-language answer using an LLM.

Responsibilities:
  - Combine retrieved code chunks, graph insights, and parser metadata
  - Generate a coherent, accurate, developer-friendly response
  - Support refactor suggestion mode
  - Track response latency
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from utils import llm, prompts

logger = logging.getLogger(__name__)

_REFACTOR_SYSTEM = """\
You are a senior software architect performing a code review.
Given the retrieved code context, suggest specific, actionable refactoring
improvements. Address: naming, structure, coupling, testability, and
documentation. Use concrete code examples where helpful. Format with markdown.
"""

_REFACTOR_USER_TEMPLATE = """\
## User Request
{query}

## Code Context
{retrieved_chunks}

## Parser Metadata
{parser_context}
"""


@dataclass
class ReasoningResult:
    """Output of the Reasoning agent."""
    answer: str
    latency_ms: float
    model_used: str
    token_estimate: int = 0
    citations: list[dict] | None = None


class ReasoningAgent:
    """
    Synthesises all upstream context into a final LLM-generated answer.

    Two modes:
      - Standard: answers questions about the codebase
      - Refactor: suggests code improvements
    """

    def __init__(self, model: str | None = None):
        import config
        self._model = model or config.LLM_MODEL

    def reason(
        self,
        query: str,
        retrieved_chunks: str,
        graph_context: str = "",
        parser_context: str = "",
        past_context: str = "",
        focus_file: str = "",
        citations: list[dict] | None = None,
        api_key: str | None = None,
    ) -> ReasoningResult:
        """
        Generate the final answer by synthesising all context.

        Args:
            query:            Original user question.
            retrieved_chunks: Formatted retrieval results.
            graph_context:    Dependency insights from GraphAgent.
            parser_context:   AST metadata from ParserAgent.
            past_context:     Relevant past Q&A from MemoryAgent.

        Returns:
            A ReasoningResult with the answer string and metrics.
        """
        is_refactor = self._is_refactor_query(query)

        if is_refactor:
            answer, latency = self._refactor_answer(query, retrieved_chunks, parser_context, api_key=api_key)
        else:
            answer, latency = self._standard_answer(
                query,
                retrieved_chunks,
                graph_context,
                parser_context,
                past_context,
                focus_file,
                api_key=api_key,
            )

        return ReasoningResult(
            answer=answer,
            latency_ms=latency,
            model_used=self._model,
            token_estimate=len(answer.split()) * 4 // 3,  # rough estimate
            citations=citations or [],
        )

    def stream_reason(
        self,
        query: str,
        retrieved_chunks: str,
        graph_context: str = "",
        parser_context: str = "",
        past_context: str = "",
        focus_file: str = "",
        api_key: str | None = None,
    ):
        is_refactor = self._is_refactor_query(query)
        if is_refactor:
            user_msg = _REFACTOR_USER_TEMPLATE.format(
                query=query,
                retrieved_chunks=retrieved_chunks,
                parser_context=parser_context or "Not available.",
            )
            yield from llm.stream_system_user(
                system=_REFACTOR_SYSTEM,
                user=user_msg,
                model=self._model,
                api_key=api_key,
            )
            return

        user_msg = prompts.reasoning_user(
            query=query,
            retrieved_chunks=retrieved_chunks,
            graph_context=graph_context,
            parser_context=parser_context,
            past_context=past_context,
            focus_file=focus_file,
        )
        yield from llm.stream_system_user(
            system=prompts.REASONING_SYSTEM,
            user=user_msg,
            model=self._model,
            api_key=api_key,
        )

    # ── private ──────────────────────────────────────────────────────────────

    def _standard_answer(
        self,
        query: str,
        retrieved_chunks: str,
        graph_context: str,
        parser_context: str,
        past_context: str,
        focus_file: str = "",
        api_key: str | None = None,
    ) -> tuple[str, float]:
        user_msg = prompts.reasoning_user(
            query=query,
            retrieved_chunks=retrieved_chunks,
            graph_context=graph_context,
            parser_context=parser_context,
            past_context=past_context,
            focus_file=focus_file,
        )
        t0 = time.perf_counter()
        answer = llm.system_user(
            system=prompts.REASONING_SYSTEM,
            user=user_msg,
            model=self._model,
            api_key=api_key,
        )
        latency = (time.perf_counter() - t0) * 1000
        logger.info("Reasoning completed in %.1fms", latency)
        return answer, latency

    def _refactor_answer(
        self,
        query: str,
        retrieved_chunks: str,
        parser_context: str,
        api_key: str | None = None,
    ) -> tuple[str, float]:
        user_msg = _REFACTOR_USER_TEMPLATE.format(
            query=query,
            retrieved_chunks=retrieved_chunks,
            parser_context=parser_context or "Not available.",
        )
        t0 = time.perf_counter()
        answer = llm.system_user(
            system=_REFACTOR_SYSTEM,
            user=user_msg,
            model=self._model,
            api_key=api_key,
        )
        latency = (time.perf_counter() - t0) * 1000
        logger.info("Refactor reasoning completed in %.1fms", latency)
        return answer, latency

    @staticmethod
    def _is_refactor_query(query: str) -> bool:
        keywords = ["refactor", "improve", "rewrite", "clean up", "suggest", "how to fix"]
        q = query.lower()
        return any(kw in q for kw in keywords)
