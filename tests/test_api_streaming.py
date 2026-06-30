import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agents.orchestrator import QueryResponse
from api.server import app


class _FakeOrchestrator:
    def prepare_query(self, query):
        context = SimpleNamespace(
            retrieved_files=["main.py"],
            citations=[{"file_path": "main.py", "start_line": 42, "label": "[source: main.py:42]"}],
        )
        return SimpleNamespace(
            query=query,
            plan=["retriever", "reasoning", "evaluator"],
            context=context,
            started_at=0.0,
            prep_ms=5.0,
        )

    def stream_prepared_answer(self, _prepared):
        yield "Streaming "
        yield "answer "
        yield "[source: main.py:42]"

    def complete_query(self, prepared, answer_text, reasoning_ms):
        return QueryResponse(
            query=prepared.query,
            answer=answer_text,
            plan=prepared.plan,
            evaluation=None,
            metrics={"reasoning_ms": reasoning_ms, "total_ms": 15.0, "chunks_searched": 1},
            retrieved_files=prepared.context.retrieved_files,
            repo_name="demo",
            citations=prepared.context.citations,
        )


class ApiStreamingTests(unittest.TestCase):
    def test_query_endpoint_streams_answer_deltas_and_final_payload(self):
        client = TestClient(app)
        with patch("api.server._get_active_orchestrator", return_value=_FakeOrchestrator()):
            with client.stream("POST", "/api/query", json={"query": "Explain it"}) as response:
                body = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: answer_start", body)
        self.assertIn("event: answer_delta", body)
        self.assertIn("Streaming answer", body)
        self.assertIn("[source: main.py:42]", body)

        payload_lines = [line for line in body.splitlines() if line.startswith("data: ")]
        final_payload = json.loads(payload_lines[-1][6:])
        self.assertEqual(final_payload["text"], "Streaming answer [source: main.py:42]")
        self.assertEqual(final_payload["citations"][0]["file_path"], "main.py")
