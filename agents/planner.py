"""
agents/planner.py — Planner Agent

Responsibilities:
  - Classify the user query into a type (architecture, dependency, refactor, etc.)
  - Decide which downstream agents to invoke and in what order
  - Return a structured execution plan consumed by the Orchestrator

The planner is the only agent that sees the raw user query first; all others
receive pre-processed inputs from the orchestrator based on the plan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from utils import llm, prompts

logger = logging.getLogger(__name__)

# Canonical query categories mapped to default agent pipelines
_DEFAULT_PLANS: dict[str, list[str]] = {
    "architecture":  ["retriever", "graph", "reasoning", "evaluator"],
    "dependency":    ["graph", "retriever", "reasoning", "evaluator"],
    "location":      ["retriever", "parser", "reasoning", "evaluator"],
    "refactor":      ["retriever", "parser", "reasoning", "evaluator"],
    "impact":        ["graph", "retriever", "reasoning", "evaluator"],
    "general":       ["retriever", "reasoning", "evaluator"],
}


@dataclass
class ExecutionPlan:
    """The output of the Planner agent."""
    agents: list[str]           # ordered list of agent names to invoke
    rationale: str              # one-line explanation
    query_type: str = "general" # inferred category
    include_memory: bool = True # whether to query memory first


class PlannerAgent:
    """
    Uses an LLM to produce an ordered agent execution plan for a given query.

    Falls back to a heuristic plan if the LLM call fails or returns
    malformed JSON.
    """

    def plan(self, query: str, history_summary: str = "", api_key: str | None = None) -> ExecutionPlan:
        """
        Determine the agent execution plan for *query*.

        Args:
            query:           The user's raw question.
            history_summary: Brief summary of recent conversation (if any).
            api_key:         Optional OpenAI API key for this request.

        Returns:
            An ExecutionPlan instance.
        """
        user_msg = prompts.planner_user(query, history_summary)
        try:
            raw = llm.system_user(
                system=prompts.PLANNER_SYSTEM,
                user=user_msg,
                max_tokens=256,
                temperature=0.0,
                api_key=api_key,
            )
            plan_data = json.loads(self._extract_json(raw))
            agents = plan_data.get("plan", [])
            rationale = plan_data.get("rationale", "LLM-generated plan")

            # Validate agent names
            valid_agents = {"retriever", "parser", "graph", "tool", "reasoning", "evaluator", "memory"}
            agents = [a for a in agents if a in valid_agents]

            if not agents:
                raise ValueError("Empty plan returned by LLM")

            # Ensure evaluator is always last
            if "evaluator" in agents:
                agents = [a for a in agents if a != "evaluator"] + ["evaluator"]

            logger.info("Plan: %s | Rationale: %s", agents, rationale)
            return ExecutionPlan(
                agents=agents,
                rationale=rationale,
                include_memory="memory" in agents,
            )

        except Exception as exc:
            logger.warning("Planner LLM failed (%s), using heuristic fallback.", exc)
            return self._heuristic_plan(query)

    # ── private ──────────────────────────────────────────────────────────────

    def _heuristic_plan(self, query: str) -> ExecutionPlan:
        """Rule-based fallback plan based on keyword detection."""
        q = query.lower()
        if any(w in q for w in ["depend", "import", "uses", "impact", "downstream"]):
            key = "dependency"
        elif any(w in q for w in ["architect", "overview", "structure", "layout"]):
            key = "architecture"
        elif any(w in q for w in ["where", "find", "located", "implemented", "defined"]):
            key = "location"
        elif any(w in q for w in ["refactor", "improve", "rewrite", "clean", "suggest"]):
            key = "refactor"
        elif any(w in q for w in ["if i modify", "change", "what happens"]):
            key = "impact"
        else:
            key = "general"

        agents = _DEFAULT_PLANS[key]
        return ExecutionPlan(
            agents=agents,
            rationale=f"Heuristic plan for query type '{key}'",
            query_type=key,
            include_memory=True,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences if present and return raw JSON string."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # remove first and last fence lines
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        return text
