"""
agents/graph_agent.py - Graph Agent

Builds and queries a directed dependency graph using NetworkX.

Nodes: source files (rel_path strings)
Edges: A -> B means A imports B

Capabilities:
  - Find all dependencies of a file
  - Find all reverse dependencies (who depends on this file?)
  - Compute downstream impact set when a file changes
  - Detect circular dependencies
  - Identify highly-connected hub files
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path, PurePosixPath

import networkx as nx

import config
from ingestion.parser import ImportInfo, ParsedModule
from utils import llm, prompts

logger = logging.getLogger(__name__)

_GRAPH_PATH = config.DEPENDENCY_GRAPH_PATH


class GraphAgent:
    """
    Dependency graph builder and query engine.

    The graph is built from pre-parsed module data (imports) during ingestion,
    then serialised to disk. At query time it's loaded and used to answer
    dependency-related questions.
    """

    def __init__(self, graph_path: Path | None = None):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._all_files: set[str] = set()
        self._graph_path = graph_path or _GRAPH_PATH

    def build(self, parsed_modules: dict[str, ParsedModule]) -> None:
        """
        Construct the dependency graph from parsed module data.

        Args:
            parsed_modules: rel_path -> ParsedModule mapping.
        """
        self._graph = nx.DiGraph()
        normalized_modules = {
            self._normalize_rel_path(rel_path): module
            for rel_path, module in parsed_modules.items()
        }
        self._all_files = set(normalized_modules.keys())

        for rel_path in normalized_modules:
            self._graph.add_node(rel_path)

        for rel_path, module in normalized_modules.items():
            for imp in module.imports:
                for target in self._resolve_import_targets(imp, rel_path, self._all_files):
                    self._graph.add_edge(rel_path, target)

        logger.info(
            "Dependency graph: %d nodes, %d edges.",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def save(self) -> None:
        """Persist the graph to disk."""
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._graph_path, "wb") as f:
            pickle.dump(self._graph, f)
        logger.info("Graph saved to %s.", self._graph_path)

    def load(self) -> bool:
        """Load graph from disk. Returns True on success."""
        if not self._graph_path.exists():
            return False
        try:
            with open(self._graph_path, "rb") as f:
                self._graph = pickle.load(f)
            self._all_files = set(self._graph.nodes)
            logger.info("Graph loaded: %d nodes.", self._graph.number_of_nodes())
            return True
        except Exception as exc:
            logger.error("Failed to load graph: %s", exc)
            return False

    def dependencies(self, rel_path: str) -> list[str]:
        """Return all files that *rel_path* directly imports."""
        rel_path = self._normalize_rel_path(rel_path)
        return list(self._graph.successors(rel_path))

    def reverse_dependencies(self, rel_path: str) -> list[str]:
        """Return all files that import *rel_path*."""
        rel_path = self._normalize_rel_path(rel_path)
        return list(self._graph.predecessors(rel_path))

    def downstream_impact(self, rel_path: str) -> list[str]:
        """
        Return all files transitively affected if *rel_path* changes.
        Uses BFS over the reversed graph.
        """
        rel_path = self._normalize_rel_path(rel_path)
        if rel_path not in self._graph:
            return []
        rev = self._graph.reverse()
        affected = nx.descendants(rev, rel_path)
        return sorted(affected)

    def has_cycles(self) -> bool:
        return not nx.is_directed_acyclic_graph(self._graph)

    def find_cycles(self) -> list[list[str]]:
        try:
            return list(nx.simple_cycles(self._graph))
        except Exception:
            return []

    def hub_files(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the *top_n* most imported files (highest in-degree)."""
        in_degrees = sorted(
            self._graph.in_degree(), key=lambda x: x[1], reverse=True
        )
        return in_degrees[:top_n]

    def summary_stats(self) -> dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "has_cycles": self.has_cycles(),
            "isolated_files": len(list(nx.isolates(self._graph))),
        }

    def to_visualization(self) -> dict:
        nodes = []
        for node in sorted(self._graph.nodes):
            nodes.append({
                "id": node,
                "label": Path(node).name,
                "path": node,
                "degree": int(self._graph.degree(node)),
                "in_degree": int(self._graph.in_degree(node)),
                "out_degree": int(self._graph.out_degree(node)),
            })
        links = [
            {"source": src, "target": dst}
            for src, dst in self._graph.edges
        ]
        return {"nodes": nodes, "links": links, "stats": self.summary_stats()}

    def answer_query(self, query: str, api_key: str | None = None) -> str:
        """
        Use the graph + LLM to answer a dependency-related question.

        Args:
            query: Natural language query about dependencies.
            api_key: Optional OpenAI API key for this request.

        Returns:
            LLM-generated answer grounded in actual graph data.
        """
        graph_data = self._build_graph_context(query)
        try:
            answer = llm.system_user(
                system=prompts.GRAPH_AGENT_SYSTEM,
                user=prompts.graph_agent_user(query, graph_data),
                max_tokens=512,
                temperature=0.1,
                api_key=api_key,
            )
            return answer
        except Exception as exc:
            logger.error("Graph agent LLM call failed: %s", exc)
            return graph_data

    def _build_graph_context(self, query: str) -> str:
        """Assemble graph facts relevant to the query into a string."""
        lines: list[str] = []
        stats = self.summary_stats()
        lines.append(
            f"Graph stats: {stats['nodes']} files, {stats['edges']} dependency edges, "
            f"cycles={'yes' if stats['has_cycles'] else 'no'}."
        )

        hubs = self.hub_files(5)
        if hubs:
            lines.append(
                "Most depended-on files: "
                + ", ".join(f"{f} ({n})" for f, n in hubs)
            )

        mentioned = [f for f in self._all_files if f in query or Path(f).stem in query]
        for rel_path in mentioned[:3]:
            deps = self.dependencies(rel_path)
            rdeps = self.reverse_dependencies(rel_path)
            impact = self.downstream_impact(rel_path)
            lines.append(
                f"\nFile: {rel_path}\n"
                f"  Direct imports: {deps or ['none']}\n"
                f"  Imported by: {rdeps or ['none']}\n"
                f"  Downstream impact if changed: {impact or ['none']}"
            )

        return "\n".join(lines)

    @staticmethod
    def _normalize_rel_path(rel_path: str) -> str:
        return PurePosixPath(rel_path.replace("\\", "/")).as_posix()

    @classmethod
    def _resolve_import_targets(
        cls, imp: ImportInfo, importing_file: str, all_files: set[str]
    ) -> list[str]:
        """Convert an import statement to repo-relative file targets."""
        importing_path = PurePosixPath(importing_file)
        base_dir = importing_path.parent
        ext = importing_path.suffix.lower()
        module_name = (imp.module or "").strip()

        candidates: list[str] = []

        if ext in {".js", ".ts"}:
            candidates.extend(cls._js_ts_candidates(module_name, imp, base_dir))
        elif ext == ".py":
            candidates.extend(cls._python_candidates(module_name, imp, base_dir))
        elif ext == ".go":
            candidates.extend(cls._path_candidates(module_name, [".go"]))
        elif ext == ".java":
            candidates.extend(cls._path_candidates(module_name.replace(".", "/"), [".java"]))
        elif ext in {".c", ".cpp", ".h", ".hpp"}:
            candidates.extend(cls._c_family_candidates(module_name, base_dir))
        elif ext == ".rs":
            candidates.extend(cls._rust_candidates(module_name, base_dir))
        else:
            candidates.extend(cls._generic_candidates(module_name, base_dir, ext))

        if imp.is_from and ext not in {".js", ".ts", ".py"}:
            for name in imp.names:
                if name == "*":
                    continue
                candidates.extend(cls._generic_candidates(name, base_dir, ext))

        return cls._expand_candidates(candidates, all_files)

    @staticmethod
    def _relative_anchor(base_dir: PurePosixPath, level: int) -> list[str]:
        if level <= 0:
            return []
        current = base_dir
        for _ in range(max(level - 1, 0)):
            current = current.parent
        return [part for part in current.parts if part not in (".", "")]

    @classmethod
    def _python_candidates(
        cls, module_name: str, imp: ImportInfo, base_dir: PurePosixPath
    ) -> list[str]:
        anchor = cls._relative_anchor(base_dir, imp.level)
        module_parts = [p for p in module_name.split(".") if p]
        base_parts = anchor + module_parts
        candidates = cls._path_candidates("/".join(base_parts), [".py"], package_files=["__init__.py"])
        if imp.is_from:
            for name in imp.names:
                if name == "*":
                    continue
                candidates.extend(
                    cls._path_candidates("/".join(base_parts + [name]), [".py"], package_files=["__init__.py"])
                )
        return candidates

    @classmethod
    def _js_ts_candidates(
        cls, module_name: str, imp: ImportInfo, base_dir: PurePosixPath
    ) -> list[str]:
        module_name = module_name.replace("\\", "/")
        if module_name.startswith("."):
            path = cls._normalize_rel_path(str(base_dir.joinpath(module_name)))
            candidates = cls._path_candidates(path, [".ts", ".js"])
        else:
            candidates = cls._path_candidates(module_name, [".ts", ".js"])
        if imp.is_from:
            for name in imp.names:
                if name in {"*", "default"}:
                    continue
                if module_name.startswith("."):
                    named_path = cls._normalize_rel_path(str(base_dir.joinpath(module_name, name)))
                else:
                    named_path = f"{module_name}/{name}"
                candidates.extend(cls._path_candidates(named_path, [".ts", ".js"]))
        return candidates

    @classmethod
    def _c_family_candidates(cls, module_name: str, base_dir: PurePosixPath) -> list[str]:
        if not module_name:
            return []
        norm = module_name.replace("\\", "/")
        if norm.startswith("."):
            norm = cls._normalize_rel_path(str(base_dir.joinpath(norm)))
        else:
            norm = cls._normalize_rel_path(str(base_dir.joinpath(norm)))
        candidates = [norm]
        stem = PurePosixPath(norm).stem
        for ext in [".h", ".hpp", ".c", ".cpp"]:
            candidates.append(PurePosixPath(norm).with_suffix(ext).as_posix())
            candidates.append(f"{stem}{ext}")
        return candidates

    @classmethod
    def _rust_candidates(cls, module_name: str, base_dir: PurePosixPath) -> list[str]:
        if not module_name:
            return []
        parts = [p.strip() for p in module_name.split("::") if p.strip()]
        if not parts:
            return []

        if parts[0] == "crate":
            parts = parts[1:]
            anchor = []
        elif parts[0] == "super":
            ups = 0
            while ups < len(parts) and parts[ups] == "super":
                ups += 1
            current = base_dir
            for _ in range(ups):
                current = current.parent
            anchor = [p for p in current.parts if p not in (".", "")]
            parts = parts[ups:]
        elif parts[0] == "self":
            anchor = [p for p in base_dir.parts if p not in (".", "")]
            parts = parts[1:]
        else:
            anchor = []

        return cls._path_candidates("/".join(anchor + parts), [".rs"], package_files=["mod.rs"])

    @classmethod
    def _generic_candidates(
        cls, module_name: str, base_dir: PurePosixPath, ext: str
    ) -> list[str]:
        if not module_name:
            return []
        if "/" in module_name or "\\" in module_name:
            normalized = cls._normalize_rel_path(str(base_dir.joinpath(module_name)))
        else:
            normalized = module_name.replace(".", "/")
        extensions = [ext] if ext else []
        return cls._path_candidates(normalized, extensions or [".txt"])

    @staticmethod
    def _path_candidates(
        path: str, extensions: list[str], package_files: list[str] | None = None
    ) -> list[str]:
        if not path:
            return []
        package_files = package_files or ["index.ts", "index.js"]
        normalized = PurePosixPath(path).as_posix()
        stem = normalized[:-len(PurePosixPath(normalized).suffix)] if PurePosixPath(normalized).suffix else normalized
        candidates = [normalized]
        if not PurePosixPath(normalized).suffix:
            for ext in extensions:
                candidates.append(f"{normalized}{ext}")
            for package_file in package_files:
                candidates.append(f"{normalized}/{package_file}")
        candidates.append(stem)
        return candidates

    @classmethod
    def _expand_candidates(cls, candidates: list[str], all_files: set[str]) -> list[str]:
        normalized_files = sorted(cls._normalize_rel_path(f) for f in all_files)
        results: list[str] = []
        for candidate in cls._dedupe(candidates):
            norm = cls._normalize_rel_path(candidate)
            if norm in all_files:
                results.append(norm)
                continue

            # Directory imports (Go/Rust/JS index-style) may refer to a package folder.
            prefix = norm.rstrip("/") + "/"
            package_matches = [f for f in normalized_files if f.startswith(prefix)]
            if package_matches:
                results.extend(package_matches)
                continue

            # Fallback: unique basename or stem match for imports like `foo` or `foo.h`.
            name = PurePosixPath(norm).name
            stem = PurePosixPath(norm).stem
            basename_matches = [
                f for f in normalized_files
                if PurePosixPath(f).name == name or PurePosixPath(f).stem == stem
            ]
            if len(basename_matches) == 1:
                results.extend(basename_matches)

        return cls._dedupe(results)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))
