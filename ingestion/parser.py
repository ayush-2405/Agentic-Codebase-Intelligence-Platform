"""
ingestion/parser.py - Structural extraction for supported source files.

Python uses the built-in AST parser for precise metadata. Other supported
languages use lightweight regex-based heuristics so ingestion, chunking,
parser metadata, and dependency graphs still work across mixed-language repos.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    name: str
    args: list[str]
    docstring: str
    lineno: int
    calls: list[str] = field(default_factory=list)
    is_method: bool = False
    decorators: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    docstring: str
    lineno: int
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    module: str
    names: list[str]
    is_from: bool
    lineno: int
    level: int = 0


@dataclass
class ParsedModule:
    file_path: str
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    global_vars: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_summary_dict(self) -> dict:
        return {
            "file": self.file_path,
            "imports": [i.module for i in self.imports],
            "classes": [
                {
                    "name": c.name,
                    "methods": [m.name for m in c.methods],
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "args": f.args,
                    "docstring": f.docstring[:120] if f.docstring else "",
                }
                for f in self.functions
            ],
        }


class PythonASTParser:
    """Precise parser for Python files using the built-in ast module."""

    def parse(self, source: str, file_path: str) -> ParsedModule:
        module = ParsedModule(file_path=file_path)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            module.parse_error = str(exc)
            logger.warning("Syntax error in %s: %s", file_path, exc)
            return module

        self._extract_imports(tree, module)
        self._extract_top_level(tree, module)
        self._extract_global_vars(tree, module)
        return module

    def _extract_imports(self, tree: ast.Module, module: ParsedModule) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(
                        ImportInfo(
                            module=alias.name,
                            names=[],
                            is_from=False,
                            lineno=node.lineno,
                            level=0,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module.imports.append(
                    ImportInfo(
                        module=node.module or "",
                        names=[alias.name for alias in node.names],
                        is_from=True,
                        lineno=node.lineno,
                        level=node.level,
                    )
                )

    def _extract_top_level(self, tree: ast.Module, module: ParsedModule) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                module.functions.append(self._parse_function(node))
            elif isinstance(node, ast.ClassDef):
                module.classes.append(self._parse_class(node))

    def _extract_global_vars(self, tree: ast.Module, module: ParsedModule) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module.global_vars.append(target.id)

    def _parse_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False
    ) -> FunctionInfo:
        return FunctionInfo(
            name=node.name,
            args=self._extract_args(node.args),
            docstring=ast.get_docstring(node) or "",
            lineno=node.lineno,
            calls=self._extract_calls(node),
            is_method=is_method,
            decorators=[self._unparse_decorator(d) for d in node.decorator_list],
        )

    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        methods: list[FunctionInfo] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(self._parse_function(child, is_method=True))
        return ClassInfo(
            name=node.name,
            bases=[self._unparse_expr(b) for b in node.bases],
            docstring=ast.get_docstring(node) or "",
            lineno=node.lineno,
            methods=methods,
            decorators=[self._unparse_decorator(d) for d in node.decorator_list],
        )

    @staticmethod
    def _extract_args(args: ast.arguments) -> list[str]:
        result = [a.arg for a in args.args]
        if args.vararg:
            result.append(f"*{args.vararg.arg}")
        if args.kwarg:
            result.append(f"**{args.kwarg.arg}")
        return result

    @staticmethod
    def _extract_calls(node: ast.AST) -> list[str]:
        calls: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                try:
                    calls.append(ast.unparse(child.func))
                except Exception:
                    continue
        return list(dict.fromkeys(calls))[:20]

    @staticmethod
    def _unparse_decorator(node: ast.expr) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "<decorator>"

    @staticmethod
    def _unparse_expr(node: ast.expr) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"


class HeuristicCodeParser:
    """Best-effort structural parser for non-Python source files."""

    _JS_TS_IMPORT_RE = re.compile(
        r'^\s*import\s+(?:(?P<names>[\w*\s{},]+)\s+from\s+)?["\'](?P<module>[^"\']+)["\']',
        re.MULTILINE,
    )
    _JS_TS_EXPORT_RE = re.compile(
        r'^\s*export\s+.*?\s+from\s+["\'](?P<module>[^"\']+)["\']',
        re.MULTILINE,
    )
    _JS_TS_REQUIRE_RE = re.compile(
        r'require\(\s*["\'](?P<module>[^"\']+)["\']\s*\)'
    )
    _GO_IMPORT_SINGLE_RE = re.compile(r'^\s*import\s+"([^"]+)"', re.MULTILINE)
    _GO_IMPORT_BLOCK_RE = re.compile(r'^\s*import\s*\((?P<body>.*?)\)', re.MULTILINE | re.DOTALL)
    _JAVA_IMPORT_RE = re.compile(r'^\s*import\s+(?:static\s+)?([\w.*]+)\s*;', re.MULTILINE)
    _CPP_INCLUDE_RE = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.MULTILINE)
    _RUST_USE_RE = re.compile(r'^\s*use\s+([^;]+);', re.MULTILINE)
    _RUST_MOD_RE = re.compile(r'^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;', re.MULTILINE)
    _GLOBAL_ASSIGN_RE = re.compile(r"^\s*(?:const|let|var|static|final|pub\s+static|type)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

    def parse(self, source: str, file_path: str) -> ParsedModule:
        module = ParsedModule(file_path=file_path)
        ext = Path(file_path).suffix.lower()

        try:
            self._extract_imports(source, module, ext)
            self._extract_classes(source, module, ext)
            self._extract_functions(source, module, ext)
            self._extract_globals(source, module)
        except Exception as exc:
            module.parse_error = str(exc)
            logger.warning("Heuristic parse issue in %s: %s", file_path, exc)

        return module

    def _extract_imports(self, source: str, module: ParsedModule, ext: str) -> None:
        if ext in {".js", ".ts"}:
            for match in self._JS_TS_IMPORT_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group("module"),
                        names=self._split_import_names(match.group("names")),
                        is_from=True,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            for match in self._JS_TS_EXPORT_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group("module"),
                        names=[],
                        is_from=True,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            for match in self._JS_TS_REQUIRE_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group("module"),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            return

        if ext == ".go":
            for match in self._GO_IMPORT_SINGLE_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group(1),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            for match in self._GO_IMPORT_BLOCK_RE.finditer(source):
                for path in re.findall(r'"([^"]+)"', match.group("body")):
                    module.imports.append(
                        ImportInfo(
                            module=path,
                            names=[],
                            is_from=False,
                            lineno=self._lineno(source, match.start()),
                        )
                    )
            return

        if ext == ".java":
            for match in self._JAVA_IMPORT_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group(1),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            return

        if ext in {".c", ".cpp", ".h", ".hpp"}:
            for match in self._CPP_INCLUDE_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group(1),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            return

        if ext == ".rs":
            for match in self._RUST_USE_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group(1).strip(),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )
            for match in self._RUST_MOD_RE.finditer(source):
                module.imports.append(
                    ImportInfo(
                        module=match.group(1),
                        names=[],
                        is_from=False,
                        lineno=self._lineno(source, match.start()),
                    )
                )

    def _extract_classes(self, source: str, module: ParsedModule, ext: str) -> None:
        patterns: list[tuple[re.Pattern[str], str]] = []
        if ext in {".js", ".ts"}:
            patterns = [
                (re.compile(r"^\s*(?:export\s+default\s+|export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "class"),
                (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "interface"),
                (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE), "type"),
            ]
        elif ext == ".go":
            patterns = [
                (re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b", re.MULTILINE), "struct"),
                (re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+interface\b", re.MULTILINE), "interface"),
            ]
        elif ext == ".java":
            patterns = [
                (re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+)*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "class"),
                (re.compile(r"^\s*(?:public\s+|private\s+|protected\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "interface"),
                (re.compile(r"^\s*(?:public\s+|private\s+|protected\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "enum"),
                (re.compile(r"^\s*(?:public\s+|private\s+|protected\s+)?record\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "record"),
            ]
        elif ext in {".c", ".cpp", ".h", ".hpp"}:
            patterns = [
                (re.compile(r"^\s*(?:typedef\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "struct"),
                (re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "class"),
            ]
        elif ext == ".rs":
            patterns = [
                (re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "struct"),
                (re.compile(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "enum"),
                (re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE), "trait"),
            ]

        for pattern, base in patterns:
            for match in pattern.finditer(source):
                module.classes.append(
                    ClassInfo(
                        name=match.group(1),
                        bases=[base],
                        docstring="",
                        lineno=self._lineno(source, match.start()),
                    )
                )

    def _extract_functions(self, source: str, module: ParsedModule, ext: str) -> None:
        patterns: list[re.Pattern[str]] = []
        if ext in {".js", ".ts"}:
            patterns = [
                re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.MULTILINE),
                re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", re.MULTILINE),
                re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?function\s*\(([^)]*)\)", re.MULTILINE),
            ]
        elif ext == ".go":
            patterns = [
                re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.MULTILINE),
            ]
        elif ext == ".java":
            patterns = [
                re.compile(r"^\s*(?:public|private|protected|static|final|native|synchronized|abstract|\s)+[\w<>\[\], ?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{", re.MULTILINE),
            ]
        elif ext in {".c", ".cpp", ".h", ".hpp"}:
            patterns = [
                re.compile(r"^\s*(?:static\s+|inline\s+|extern\s+|virtual\s+|constexpr\s+)*[\w:\<\>\*\&\s]+\s+([A-Za-z_][A-Za-z0-9_:~]*)\s*\(([^;{}()]*)\)\s*\{", re.MULTILINE),
            ]
        elif ext == ".rs":
            patterns = [
                re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]+>)?\s*\(([^)]*)\)", re.MULTILINE),
            ]

        for pattern in patterns:
            for match in pattern.finditer(source):
                name = match.group(1)
                args = self._split_args(match.group(2))
                module.functions.append(
                    FunctionInfo(
                        name=name,
                        args=args,
                        docstring="",
                        lineno=self._lineno(source, match.start()),
                    )
                )

        module.functions = self._dedupe_functions(module.functions)

    def _extract_globals(self, source: str, module: ParsedModule) -> None:
        module.global_vars = list(dict.fromkeys(m.group(1) for m in self._GLOBAL_ASSIGN_RE.finditer(source)))[:20]

    @staticmethod
    def _lineno(source: str, offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    @staticmethod
    def _split_import_names(raw: str | None) -> list[str]:
        if not raw:
            return []
        cleaned = raw.replace("{", "").replace("}", "").replace("* as", "*").replace(",", " ")
        return [part for part in cleaned.split() if part not in {"as", "type", "default"}][:12]

    @staticmethod
    def _split_args(raw: str) -> list[str]:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        cleaned: list[str] = []
        for part in parts:
            name = part.split(":")[0].split("=")[0].strip()
            if name:
                cleaned.append(name)
        return cleaned[:20]

    @staticmethod
    def _dedupe_functions(functions: list[FunctionInfo]) -> list[FunctionInfo]:
        seen: set[tuple[str, int]] = set()
        result: list[FunctionInfo] = []
        for fn in functions:
            key = (fn.name, fn.lineno)
            if key in seen:
                continue
            seen.add(key)
            result.append(fn)
        return result


_python_parser = PythonASTParser()
_heuristic_parser = HeuristicCodeParser()


def parse_python_file(source: str, file_path: str) -> ParsedModule:
    return _python_parser.parse(source, file_path)


def parse_source_file(source: str, file_path: str) -> ParsedModule:
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return parse_python_file(source, file_path)
    return _heuristic_parser.parse(source, file_path)


def parse_file_from_disk(path: Path) -> ParsedModule:
    source = path.read_text(encoding="utf-8", errors="replace")
    return parse_source_file(source, str(path))
