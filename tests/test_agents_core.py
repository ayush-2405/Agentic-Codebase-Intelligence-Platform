import unittest
from unittest.mock import patch

from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from agents.reasoning_agent import ReasoningAgent


class PlannerTests(unittest.TestCase):
    def test_extract_json_strips_markdown_fences(self):
        raw = "```json\n{\"plan\": [\"retriever\"], \"rationale\": \"ok\"}\n```"
        self.assertEqual(PlannerAgent._extract_json(raw), "{\"plan\": [\"retriever\"], \"rationale\": \"ok\"}")

    def test_plan_uses_llm_response_and_moves_evaluator_last(self):
        with patch("agents.planner.llm.system_user", return_value='{"plan":["memory","retriever","evaluator","graph"],"rationale":"x"}'):
            plan = PlannerAgent().plan("how does this work?")
        self.assertEqual(plan.agents, ["memory", "retriever", "graph", "evaluator"])
        self.assertTrue(plan.include_memory)

    def test_plan_falls_back_to_heuristics(self):
        with patch("agents.planner.llm.system_user", side_effect=RuntimeError("boom")):
            plan = PlannerAgent().plan("where is auth implemented")
        self.assertIn("parser", plan.agents)
        self.assertEqual(plan.query_type, "location")


class EvaluatorTests(unittest.TestCase):
    def test_evaluate_parses_result(self):
        raw = '{"correctness_score":0.9,"completeness_score":0.8,"clarity_score":0.7,"overall_score":0.8,"issues":["minor"],"suggested_improvements":"none"}'
        with patch("agents.evaluator.llm.system_user", return_value=raw):
            result = EvaluatorAgent(threshold=0.75).evaluate("q", "a")
        self.assertTrue(result.passed)
        self.assertEqual(result.issues, ["minor"])

    def test_evaluate_fallback_on_failure(self):
        with patch("agents.evaluator.llm.system_user", side_effect=RuntimeError("fail")):
            result = EvaluatorAgent().evaluate("q", "a")
        self.assertTrue(result.passed)
        self.assertIn("Evaluation could not be performed.", result.issues)


class ReasoningTests(unittest.TestCase):
    def test_reason_calls_standard_llm_path(self):
        with patch("agents.reasoning_agent.llm.system_user", return_value="answer"):
            result = ReasoningAgent(model="test-model").reason("question", "chunks")
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.model_used, "test-model")

    def test_stream_reason_yields_deltas(self):
        with patch("agents.reasoning_agent.llm.stream_system_user", return_value=iter(["a", "b", "c"])):
            deltas = list(ReasoningAgent().stream_reason("question", "chunks"))
        self.assertEqual(deltas, ["a", "b", "c"])

    def test_is_refactor_query_detects_keywords(self):
        self.assertTrue(ReasoningAgent._is_refactor_query("please refactor this"))
        self.assertFalse(ReasoningAgent._is_refactor_query("where is the router"))
