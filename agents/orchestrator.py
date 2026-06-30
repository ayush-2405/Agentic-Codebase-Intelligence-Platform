"""
agents/orchestrator.py - Orchestrator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents.evaluator import EvaluationResult, EvaluatorAgent
from agents.graph_agent import GraphAgent
from agents.memory_agent import MemoryAgent, MemoryEntry
from agents.parser_agent import ParserAgent
from agents.planner import ExecutionPlan, PlannerAgent
from agents.reasoning_agent import ReasoningAgent
from agents.retriever import RetrieverAgent
from agents.tool_agent import ToolAgent
from ingestion.indexer import VectorIndex, get_index
from ingestion.parser import ParsedModule
import config

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    query: str
    answer: str
    plan: list[str]
    evaluation: EvaluationResult | None
    metrics: dict = field(default_factory=dict)
    retrieved_files: list[str] = field(default_factory=list)
    error: str | None = None
    repo_name: str = ""
    citations: list[dict] = field(default_factory=list)

    def display(self) -> str:
        lines = [
            "=" * 70,
            f"QUERY: {self.query}",
            "=" * 70,
            "",
            self.answer,
            "",
            "=" * 70,
            f"PLAN: {' -> '.join(self.plan)}",
        ]
        if self.evaluation:
            lines.append(f"EVAL: {self.evaluation.summary()}")
        if self.metrics:
            m = self.metrics
            lines.append(
                f"METRICS: total={m.get('total_ms', 0):.0f}ms | "
                f"retrieval={m.get('retrieval_ms', 0):.0f}ms | "
                f"reasoning={m.get('reasoning_ms', 0):.0f}ms | "
                f"chunks_searched={m.get('chunks_searched', 0)}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class _AgentContext:
    query: str
    past_context: str = ""
    retrieved_chunks: str = ""
    retrieved_files: list[str] = field(default_factory=list)
    focus_file: str | None = None
    parser_context: str = ""
    graph_context: str = ""
    tool_context: str = ""
    answer: str = ""
    evaluation: EvaluationResult | None = None
    chunks_searched: int = 0
    citations: list[dict] = field(default_factory=list)


@dataclass
class PreparedQuery:
    query: str
    plan: list[str]
    context: _AgentContext
    started_at: float
    prep_ms: float
    retrieval_ms: float
    api_key: str | None = None


class Orchestrator:
    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        parsed_modules: dict[str, ParsedModule] | None = None,
        graph_agent: GraphAgent | None = None,
        repo_path: Path | None = None,
        history_path: Path | None = None,
        repo_name: str = "",
    ):
        self._planner = PlannerAgent()
        self._retriever = RetrieverAgent(index=vector_index or get_index())
        self._parser_agent = ParserAgent(parsed_cache=parsed_modules)
        self._graph_agent = graph_agent or GraphAgent()
        self._repo_path = repo_path or config.DATA_DIR
        self._repo_name = repo_name or self._repo_path.name
        self._tool_agent = ToolAgent(root_dir=self._repo_path)
        self._reasoning_agent = ReasoningAgent()
        self._evaluator = EvaluatorAgent()
        self._memory = MemoryAgent(history_path=history_path)
        logger.info("Orchestrator initialised.")

    def query(self, user_query: str, api_key: str | None = None) -> QueryResponse:
        prepared = self.prepare_query(user_query, api_key=api_key)
        reasoning_result = self._reasoning_agent.reason(
            query=user_query,
            retrieved_chunks=prepared.context.retrieved_chunks or "No context retrieved.",
            graph_context=prepared.context.graph_context,
            parser_context=prepared.context.parser_context,
            past_context=prepared.context.past_context,
            citations=prepared.context.citations,
            api_key=prepared.api_key,
        )
        return self.complete_query(prepared, reasoning_result.answer, reasoning_result.latency_ms)

    def prepare_query(self, user_query: str, focus_file: str | None = None, api_key: str | None = None) -> PreparedQuery:
        t_start = time.perf_counter()
        logger.info("Processing query: %.120s", user_query)

        past_entries = self._memory.retrieve_similar(user_query)
        past_context = self._memory.format_past_context(past_entries)
        history_summary = self._memory.recent_summary()
        plan: ExecutionPlan = self._planner.plan(user_query, history_summary, api_key=api_key)
        context = _AgentContext(query=user_query, past_context=past_context, focus_file=focus_file)

        retrieval_ms = 0.0
        focus_context = ""
        focus_retrieved_files: list[str] = []
        if focus_file:
            focus_chunks = self._retriever._index.get_chunk_by_file(focus_file) if self._retriever._index else []
            if focus_chunks:
                focus_context = self._retriever.format_chunks(focus_chunks[:8], header=focus_file)
                focus_retrieved_files = [focus_file]

        for agent_name in plan.agents:
            if agent_name == "retriever":
                result = self._retriever.retrieve(user_query)
                if focus_context:
                    context.retrieved_chunks = (
                        focus_context + "\n\n---\n\n" + result.formatted_context
                        if result.formatted_context else focus_context
                    )
                    context.retrieved_files = list(dict.fromkeys(focus_retrieved_files + [r.chunk.file_path for r in result.search_results]))
                else:
                    context.retrieved_chunks = result.formatted_context
                    context.retrieved_files = [r.chunk.file_path for r in result.search_results]
                context.chunks_searched = result.num_chunks_searched
                context.citations = result.citations
                retrieval_ms = result.latency_ms
            elif agent_name == "parser":
                files_to_parse = list(dict.fromkeys(context.retrieved_files))[:5]
                context.parser_context = self._parser_agent.get_metadata(files_to_parse)
            elif agent_name == "graph":
                context.graph_context = self._graph_agent.answer_query(user_query, api_key=api_key)
            elif agent_name == "tool":
                tool_results = self._tool_agent.search_repo(user_query[:50], max_results=10)
                if tool_results:
                    context.tool_context = self._tool_agent.format_search_results(tool_results)
            elif agent_name in {"reasoning", "evaluator", "memory"}:
                continue

        return PreparedQuery(
            query=user_query,
            plan=plan.agents,
            context=context,
            started_at=t_start,
            prep_ms=(time.perf_counter() - t_start) * 1000,
            retrieval_ms=retrieval_ms,
            api_key=api_key,
        )

    def stream_prepared_answer(self, prepared: PreparedQuery, api_key: str | None = None):
        yield from self._reasoning_agent.stream_reason(
            query=prepared.query,
            retrieved_chunks=prepared.context.retrieved_chunks or "No context retrieved.",
            graph_context=prepared.context.graph_context,
            parser_context=prepared.context.parser_context,
            past_context=prepared.context.past_context,
            focus_file=prepared.context.focus_file or "",
            api_key=api_key,
        )

    def complete_query(self, prepared: PreparedQuery, answer_text: str, reasoning_ms: float) -> QueryResponse:
        prepared.context.answer = answer_text
        if "evaluator" in prepared.plan and prepared.context.answer:
            prepared.context.evaluation = self._evaluator.evaluate(
                prepared.query,
                prepared.context.answer,
                api_key=prepared.api_key,
            )

        total_ms = (time.perf_counter() - prepared.started_at) * 1000
        self._memory.store(
            MemoryEntry(
                query=prepared.query,
                answer=prepared.context.answer,
                evaluation_score=(
                    prepared.context.evaluation.overall_score if prepared.context.evaluation else 0.0
                ),
                agent_plan=prepared.plan,
                retrieved_files=prepared.context.retrieved_files,
                latency_ms=total_ms,
            )
        )

        return QueryResponse(
            query=prepared.query,
            answer=prepared.context.answer,
            plan=prepared.plan,
            evaluation=prepared.context.evaluation,
            metrics={
                "total_ms": total_ms,
                "prep_ms": prepared.prep_ms,
                "retrieval_ms": prepared.retrieval_ms,
                "reasoning_ms": reasoning_ms,
                "chunks_searched": prepared.context.chunks_searched,
            },
            retrieved_files=prepared.context.retrieved_files,
            repo_name=self._repo_name,
            citations=prepared.context.citations,
        )

    def analyze_diff(self, diff_text: str, user_query: str) -> QueryResponse:
        t_start = time.perf_counter()
        files = _extract_changed_files(diff_text)
        changed_files = [f for f in files if f]

        diff_context_parts = ["## Proposed Diff", diff_text[:12000]]
        retrieved_files: list[str] = []

        if changed_files:
            retrieved_files = [f for f in changed_files if f in self._tool_agent.list_files()]
            impacted = []
            for rel_path in retrieved_files[:10]:
                deps = self._graph_agent.dependencies(rel_path)
                rdeps = self._graph_agent.reverse_dependencies(rel_path)
                impacted.append(
                    f"- {rel_path}\n"
                    f"  imports: {deps or ['none']}\n"
                    f"  imported by: {rdeps or ['none']}\n"
                    f"  downstream impact: {self._graph_agent.downstream_impact(rel_path) or ['none']}"
                )
            if impacted:
                diff_context_parts.append("## Graph Impact\n" + "\n".join(impacted))

            chunks = []
            index = self._retriever._index
            for rel_path in retrieved_files[:8]:
                chunks.extend(index.get_chunk_by_file(rel_path)[:2])
            if chunks:
                chunk_text = "\n\n".join(
                    f"### {chunk.file_path}\n```\n{chunk.content[:1800]}\n```"
                    for chunk in chunks[:10]
                )
                diff_context_parts.append("## Existing Code Context\n" + chunk_text)

            parser_context = self._parser_agent.get_metadata(retrieved_files[:6])
        else:
            parser_context = ""

        augmented_query = (
            f"{user_query}\n\n"
            "Treat the pasted diff as a proposed change that has not yet been applied. "
            "Explain likely breakages, risky dependencies, tests to run, and rollout concerns."
        )

        reasoning_result = self._reasoning_agent.reason(
            query=augmented_query,
            retrieved_chunks="\n\n".join(diff_context_parts),
            graph_context=self._graph_agent.answer_query(
                f"What breaks if these files change? {', '.join(retrieved_files[:8])}"
            ) if retrieved_files else "",
            parser_context=parser_context,
        )
        evaluation = self._evaluator.evaluate(user_query, reasoning_result.answer)
        total_ms = (time.perf_counter() - t_start) * 1000
        self._memory.store(
            MemoryEntry(
                query=f"[DIFF] {user_query}",
                answer=reasoning_result.answer,
                evaluation_score=evaluation.overall_score if evaluation else 0.0,
                agent_plan=["diff", "graph", "reasoning", "evaluator"],
                retrieved_files=retrieved_files,
                latency_ms=total_ms,
            )
        )
        return QueryResponse(
            query=user_query,
            answer=reasoning_result.answer,
            plan=["diff", "graph", "reasoning", "evaluator"],
            evaluation=evaluation,
            metrics={"total_ms": total_ms, "chunks_searched": self._retriever._index.num_chunks},
            retrieved_files=retrieved_files,
            repo_name=self._repo_name,
        )

    @property
    def memory(self) -> MemoryAgent:
        return self._memory

    @property
    def graph(self) -> GraphAgent:
        return self._graph_agent

    @property
    def tools(self) -> ToolAgent:
        return self._tool_agent

    @property
    def repo_name(self) -> str:
        return self._repo_name


def _extract_changed_files(diff_text: str) -> list[str]:
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    files.append(path[2:])
    return list(dict.fromkeys(files))
