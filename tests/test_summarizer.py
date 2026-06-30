import asyncio
import time
import unittest
from unittest.mock import patch

from ingestion.loader import CodeFile
from ingestion.summarizer import summarize_files_async


class SummarizerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_summarization_respects_concurrency(self):
        files = [
            CodeFile(path=None, rel_path=f"file_{idx}.py", extension=".py", content="print('x')")  # type: ignore[arg-type]
            for idx in range(4)
        ]
        calls = []

        def fake_summarize(_self, code_file, _parsed=None):
            calls.append(code_file.rel_path)
            time.sleep(0.12)
            return f"summary:{code_file.rel_path}"

        started = time.perf_counter()
        with patch("ingestion.summarizer.FileSummarizer.summarize", new=fake_summarize):
            summaries = await summarize_files_async(files, max_concurrency=4)
        elapsed = time.perf_counter() - started

        self.assertEqual(set(summaries), {f.rel_path for f in files})
        self.assertLess(elapsed, 0.32, "expected concurrent summarization to complete faster than sequential execution")
        self.assertEqual(len(calls), 4)
