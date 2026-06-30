"""
agents/evaluator.py — Evaluator Agent

LLM-based quality gate for generated answers. Scores answers on:
  - Correctness    (does it answer the actual question?)
  - Completeness   (does it cover all relevant aspects?)
  - Clarity        (is it well-structured and readable?)

If the overall score falls below the threshold, the evaluator flags issues
and suggests improvements that the orchestrator can use to retry or annotate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from utils import llm, prompts
import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured output of the Evaluator agent."""
    correctness_score: float
    completeness_score: float
    clarity_score: float
    overall_score: float
    issues: list[str] = field(default_factory=list)
    suggested_improvements: str = ""
    passed: bool = False  # True if overall_score ≥ MIN_ACCEPTABLE_SCORE
    raw_response: str = ""

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "✅ PASS" if self.passed else "⚠️  NEEDS IMPROVEMENT"
        return (
            f"{status} | Overall: {self.overall_score:.2f} "
            f"(correctness={self.correctness_score:.2f}, "
            f"completeness={self.completeness_score:.2f}, "
            f"clarity={self.clarity_score:.2f})"
        )


class EvaluatorAgent:
    """
    Evaluates the quality of a Reasoning agent answer using an LLM judge.

    Implements the LLM-as-judge pattern with structured JSON output.
    Falls back gracefully if the LLM returns malformed JSON.
    """

    def __init__(self, threshold: float | None = None):
        self._threshold = threshold if threshold is not None else config.MIN_ACCEPTABLE_SCORE

    def evaluate(self, query: str, answer: str, api_key: str | None = None) -> EvaluationResult:
        """
        Score *answer* against the original *query*.

        Args:
            query:  The original user question.
            answer: The generated answer to evaluate.
            api_key: Optional OpenAI API key for this request.

        Returns:
            An EvaluationResult with scores and improvement suggestions.
        """
        if not config.EVALUATION_ENABLED:
            return self._skip_result()

        user_msg = prompts.evaluator_user(query, answer)
        try:
            raw = llm.system_user(
                system=prompts.EVALUATOR_SYSTEM,
                user=user_msg,
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
                api_key=api_key,
            )
            result = self._parse_result(raw)
            result.passed = result.overall_score >= self._threshold
            logger.info("Evaluation: %s", result.summary())
            return result
        except Exception as exc:
            logger.warning("Evaluator failed: %s. Using fallback.", exc)
            return self._fallback_result()

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(raw: str) -> EvaluationResult:
        data = json.loads(raw)
        return EvaluationResult(
            correctness_score=float(data.get("correctness_score", 0.5)),
            completeness_score=float(data.get("completeness_score", 0.5)),
            clarity_score=float(data.get("clarity_score", 0.5)),
            overall_score=float(data.get("overall_score", 0.5)),
            issues=data.get("issues", []),
            suggested_improvements=data.get("suggested_improvements", ""),
            raw_response=raw,
        )

    @staticmethod
    def _fallback_result() -> EvaluationResult:
        """Neutral result when evaluation cannot be performed."""
        return EvaluationResult(
            correctness_score=0.5,
            completeness_score=0.5,
            clarity_score=0.5,
            overall_score=0.5,
            issues=["Evaluation could not be performed."],
            suggested_improvements="",
            passed=True,  # Don't block on evaluator failures
        )

    @staticmethod
    def _skip_result() -> EvaluationResult:
        return EvaluationResult(
            correctness_score=1.0,
            completeness_score=1.0,
            clarity_score=1.0,
            overall_score=1.0,
            passed=True,
        )
