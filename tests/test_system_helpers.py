import json
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import numpy as np

import config
from agents.graph_agent import GraphAgent
from agents.memory_agent import MemoryAgent, MemoryEntry
from agents.parser_agent import ParserAgent
from agents.tool_agent import ToolAgent
from api import repo_store
from ingestion.chunker import CodeChunker
from ingestion.loader import RepoLoader
from ingestion.manifest import IngestManifest, IngestedFile
from ingestion.parser import ParsedModule, parse_source_file
from main import _apply_repo_to_config, build_full_parser
from utils.embeddings import cosine_similarity


class RepoFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(f"tests/_tmp_system_{uuid4().hex}").resolve()
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pkg").mkdir(exist_ok=True)
        (self.root / "a.py").write_text("import os\n\nCONST = 1\n\ndef alpha(x):\n    return x + 1\n", encoding="utf-8")
        (self.root / "pkg" / "b.js").write_text("import x from './c'\nexport function beta(y) { return y }\n", encoding="utf-8")
        (self.root / "pkg" / "c.js").write_text("export class Gamma {}\n", encoding="utf-8")
        (self.root / ".hidden.py").write_text("print('skip')", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class LoaderParserChunkerTests(RepoFixture):
    def test_loader_skips_hidden_and_hashes_content(self):
        files = RepoLoader(root_dir=self.root).load()
        rels = {f.rel_path for f in files}
        self.assertIn("a.py", rels)
        self.assertIn("pkg\\b.js", rels)
        self.assertNotIn(".hidden.py", rels)
        self.assertTrue(all(f.metadata.get("sha256") for f in files))

    def test_parse_source_file_supports_python_and_js(self):
        py = parse_source_file((self.root / "a.py").read_text(encoding="utf-8"), "a.py")
        js = parse_source_file((self.root / "pkg" / "b.js").read_text(encoding="utf-8"), "pkg/b.js")
        self.assertEqual(py.functions[0].name, "alpha")
        self.assertEqual(js.functions[0].name, "beta")

    def test_chunker_creates_definition_and_summary_chunks(self):
        files = RepoLoader(root_dir=self.root).load()
        py_file = next(f for f in files if f.rel_path == "a.py")
        parsed = parse_source_file(py_file.content, py_file.rel_path)
        chunks = CodeChunker().chunk(py_file, parsed, "summary text")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk.chunk_type == "module" for chunk in chunks))


class ToolParserMemoryTests(RepoFixture):
    def test_tool_agent_methods(self):
        tool = ToolAgent(root_dir=self.root)
        self.assertIn("alpha", tool.read_file("a.py"))
        self.assertEqual(tool.get_file_stats("a.py")["extension"], ".py")
        self.assertTrue(tool.list_files(".py"))
        results = tool.search_repo("alpha")
        self.assertEqual(results[0]["file"], "a.py")
        self.assertIn("a.py:", tool.format_search_results(results))

    def test_parser_agent_uses_cache_and_finds_symbols(self):
        parsed = parse_source_file((self.root / "a.py").read_text(encoding="utf-8"), "a.py")
        agent = ParserAgent(parsed_cache={"a.py": parsed})
        metadata = agent.get_metadata(["a.py"])
        self.assertIn("alpha", metadata)
        symbols = agent.find_symbol("alpha")
        self.assertEqual(symbols[0]["type"], "function")

    def test_memory_agent_store_retrieve_and_clear(self):
        history = self.root / "memory.json"
        with patch("agents.memory_agent.embed_text", side_effect=lambda text: np.array([len(text), 1.0], dtype=np.float32)):
            memory = MemoryAgent(history_path=history)
            memory.store(MemoryEntry(query="hello", answer="world"))
            memory.store(MemoryEntry(query="help", answer="answer"))
            similar = memory.retrieve_similar("hello", top_k=1)
        self.assertEqual(similar[0].query, "hello")
        self.assertIn("hello", memory.recent_summary())
        self.assertTrue(memory.format_past_context(similar))
        memory.clear()
        self.assertEqual(memory.count, 0)


class GraphManifestRepoStoreTests(RepoFixture):
    def test_graph_agent_build_and_visualization(self):
        parsed = {
            "a.py": parse_source_file((self.root / "a.py").read_text(encoding="utf-8"), "a.py"),
            "pkg/b.js": parse_source_file((self.root / "pkg" / "b.js").read_text(encoding="utf-8"), "pkg/b.js"),
            "pkg/c.js": parse_source_file((self.root / "pkg" / "c.js").read_text(encoding="utf-8"), "pkg/c.js"),
        }
        graph = GraphAgent(graph_path=self.root / "graph.pkl")
        graph.build(parsed)
        viz = graph.to_visualization()
        self.assertEqual(viz["stats"]["nodes"], 3)
        self.assertTrue(isinstance(viz["links"], list))

    def test_manifest_save_and_load(self):
        path = self.root / "manifest.json"
        manifest = IngestManifest(repo_path=str(self.root), files={"a.py": IngestedFile(rel_path="a.py", sha256="abc")})
        manifest.save(path)
        loaded = IngestManifest.load(path)
        self.assertEqual(loaded.files["a.py"].sha256, "abc")

    def test_repo_store_roundtrip(self):
        registry_path = self.root / "registry.json"
        with patch.object(config, "REPO_REGISTRY_PATH", registry_path), patch.object(config, "REPO_STORE_DIR", self.root / "store"):
            repo_store.upsert_repo_record({"repo_id": "r1", "repo_name": "demo", "repo_path": str(self.root), "updated_at": 1}, set_active=True)
            repo_store.set_active_repo("r1")
            record = repo_store.get_repo_record("r1")
            self.assertEqual(record["repo_name"], "demo")
            self.assertEqual(repo_store.list_repos()["active_repo_id"], "r1")


class MainEmbeddingsAndDataTests(RepoFixture):
    def test_main_parser_and_config_apply(self):
        parser = build_full_parser()
        args = parser.parse_args(["query", "--repo", str(self.root), "--ask", "hi"])
        self.assertEqual(args.command, "query")
        _apply_repo_to_config(self.root)
        self.assertEqual(config.DATA_DIR, self.root)

    def test_cosine_similarity(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

    def test_data_repo_modules(self):
        data_repo = Path("data/repo").resolve()
        if str(data_repo) not in sys.path:
            sys.path.insert(0, str(data_repo))
        from data_utils import StandardScaler, generate_synthetic_data, train_test_split
        from evaluator import ModelEvaluator
        from models import LinearRegression, ModelConfig
        from trainer import run_experiment

        X, y = generate_synthetic_data(n_samples=20, n_features=3, seed=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=2)
        scaled = StandardScaler().fit_transform(X_train)
        self.assertEqual(scaled.shape, X_train.shape)

        model = LinearRegression(ModelConfig(epochs=5))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = ModelEvaluator().compute_metrics(y_test, preds)
        self.assertIn("mse", metrics)

        result = run_experiment(ModelConfig(epochs=3), X, y)
        self.assertIn("train_time_s", result)
