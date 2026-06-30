"""
api/server.py - FastAPI backend for the Codebase Intelligence app.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agents.graph_agent import GraphAgent
from agents.orchestrator import Orchestrator
from api.repo_store import (
    find_repo_record,
    get_repo_record,
    list_repos as load_repo_registry,
    make_repo_id,
    repo_record_paths,
    set_active_repo,
    upsert_repo_record,
)
from ingestion.indexer import VectorIndex
from ingestion.pipeline import ingest_repository

logger = logging.getLogger(__name__)

app = FastAPI(title="ACIP Web", docs_url="/api/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {
    "orchestrator": None,
    "repo_id": None,
    "repo_url": None,
    "repo_name": None,
    "repo_path": None,
    "indexed": False,
    "file_count": 0,
    "chunk_count": 0,
    "graph_stats": {},
    "repos": [],
}


def _emit(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


class DiffRequest(BaseModel):
    diff: str = Field(..., min_length=5)
    query: str = Field(default="What breaks if I apply this diff?")


class SearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1)
    regex: bool = Field(default=False)
    case_sensitive: bool = Field(default=False)
    extension: str = Field(default="")
    max_results: int = Field(default=50, ge=1, le=500)


class IngestRequest(BaseModel):
    github_url: str = Field(..., description="GitHub repo URL, e.g. https://github.com/user/repo")
    branch: str = Field(default="", description="Branch name (leave blank for default)")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    focus_file: str | None = None
    openai_api_key: str | None = None


def _registry_payload() -> dict:
    registry = load_repo_registry()
    _state["repos"] = registry.get("repos", [])
    return registry


def _repo_summary(record: dict) -> dict:
    return {
        "repo_id": record.get("repo_id"),
        "repo_name": record.get("repo_name"),
        "repo_url": record.get("repo_url"),
        "repo_path": record.get("repo_path"),
        "branch": record.get("branch", ""),
        "file_count": record.get("file_count", 0),
        "chunk_count": record.get("chunk_count", 0),
        "graph_stats": record.get("graph_stats", {}),
        "updated_at": record.get("updated_at", 0),
    }


def _build_index(paths: dict) -> VectorIndex:
    return VectorIndex(
        index_path=paths["index_path"],
        chunks_path=paths["chunks_path"],
        vectors_path=paths["vectors_path"],
    )


def _activate_repo_record(record: dict) -> Orchestrator:
    paths = repo_record_paths(record["repo_id"])
    repo_path = Path(record["repo_path"])
    config.DATA_DIR = repo_path

    index = _build_index(paths)
    if not index.load():
        raise HTTPException(500, f"Missing saved index for repo {record['repo_name']}")

    graph = GraphAgent(graph_path=paths["graph_path"])
    graph.load()

    orchestrator = Orchestrator(
        vector_index=index,
        graph_agent=graph,
        repo_path=repo_path,
        history_path=paths["memory_path"],
        repo_name=record["repo_name"],
    )
    _state.update(
        {
            "orchestrator": orchestrator,
            "repo_id": record["repo_id"],
            "repo_url": record.get("repo_url"),
            "repo_name": record["repo_name"],
            "repo_path": str(repo_path),
            "indexed": True,
            "file_count": record.get("file_count", 0),
            "chunk_count": record.get("chunk_count", 0),
            "graph_stats": record.get("graph_stats", {}),
        }
    )
    _registry_payload()
    return orchestrator


def _get_active_orchestrator() -> Orchestrator:
    orch = _state.get("orchestrator")
    if orch:
        return orch
    registry = _registry_payload()
    active_repo_id = registry.get("active_repo_id")
    if not active_repo_id:
        raise HTTPException(404, "No repo indexed")
    record = get_repo_record(active_repo_id)
    if not record:
        raise HTTPException(404, "Active repo not found")
    return _activate_repo_record(record)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(title: str, lines: list[str]) -> bytes:
    page_lines = [title, ""] + lines
    y = 800
    content_lines = ["BT", "/F1 10 Tf"]
    for raw_line in page_lines:
        for piece in (raw_line or " ").splitlines() or [" "]:
            safe = _pdf_escape(piece[:110])
            content_lines.append(f"1 0 0 1 40 {y} Tm ({safe}) Tj")
            y -= 14
            if y < 50:
                break
        if y < 50:
            break
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj",
    ]

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(buffer.tell())
        buffer.write(obj + b"\n")
    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    buffer.write(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode("latin-1")
    )
    return buffer.getvalue()


def response_time_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


async def _stream_reasoning_answer(orch: Orchestrator, prepared, delta_queue: asyncio.Queue) -> str:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

    def _worker() -> None:
        try:
            for delta in orch.stream_prepared_answer(prepared, api_key=prepared.api_key):
                asyncio.run_coroutine_threadsafe(queue.put(("delta", delta)), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    chunks: list[str] = []
    while True:
        kind, payload = await queue.get()
        if kind == "delta" and payload:
            chunks.append(payload)
            await delta_queue.put(payload)
        elif kind == "error":
            raise RuntimeError(payload or "Streaming failed")
        elif kind == "done":
            break

    thread.join(timeout=0.1)
    return "".join(chunks)


@app.post("/api/ingest")
async def ingest_stream(req: IngestRequest):
    async def generate() -> AsyncGenerator:
        repo_url = req.github_url.strip().rstrip("/")
        if not repo_url.startswith("http"):
            yield _emit("error", {"message": "Invalid GitHub URL"})
            return

        display_url = repo_url[:-4] if repo_url.endswith(".git") else repo_url
        clone_url = repo_url if repo_url.endswith(".git") else repo_url + ".git"
        repo_name = display_url.rstrip("/").split("/")[-1]

        yield _emit("step", {"phase": "clone", "message": f"Cloning {repo_name}...", "progress": 5})

        clone_dir = Path(tempfile.mkdtemp(prefix="acip_"))
        try:
            import git as gitlib

            kwargs = {"depth": 1}
            if req.branch:
                kwargs["branch"] = req.branch

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, lambda: gitlib.Repo.clone_from(clone_url, clone_dir, **kwargs))
            except gitlib.GitCommandError as exc:
                yield _emit("error", {"message": f"Clone failed: {str(exc)[:200]}"})
                shutil.rmtree(clone_dir, ignore_errors=True)
                return

            yield _emit("step", {"phase": "clone", "message": "Cloned successfully", "progress": 15})

            existing = find_repo_record(display_url, req.branch)
            repo_id = existing["repo_id"] if existing else make_repo_id(repo_name)
            paths = repo_record_paths(repo_id)
            paths["base"].mkdir(parents=True, exist_ok=True)

            config.DATA_DIR = clone_dir
            _state.update(
                {
                    "repo_id": repo_id,
                    "repo_path": str(clone_dir),
                    "repo_url": display_url,
                    "repo_name": repo_name,
                    "indexed": False,
                }
            )

            index = _build_index(paths)
            graph = GraphAgent(graph_path=paths["graph_path"])
            progress_queue: asyncio.Queue[dict] = asyncio.Queue()

            async def _progress(phase: str, message: str, pct: int, extra: dict) -> None:
                payload = {"phase": phase, "message": message, "progress": pct}
                payload.update(extra)
                await progress_queue.put(_emit("step", payload))

            ingest_task = asyncio.create_task(
                ingest_repository(
                    repo_path=clone_dir,
                    index=index,
                    graph=graph,
                    manifest_path=paths["manifest_path"],
                    progress=_progress,
                )
            )

            while True:
                if ingest_task.done() and progress_queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield event

            result = await ingest_task

            if not result.files:
                yield _emit(
                    "error",
                    {"message": f"No supported source files found. Extensions searched: {config.SUPPORTED_EXTENSIONS}"},
                )
                return

            snapshot_record = {
                "repo_id": repo_id,
                "repo_name": repo_name,
                "repo_url": display_url,
                "repo_path": str(clone_dir),
                "branch": req.branch,
                "file_count": len(result.files),
                "chunk_count": len(result.chunks),
                "graph_stats": result.graph_stats,
                "updated_at": time.time(),
            }
            paths["meta_path"].write_text(json.dumps(snapshot_record, indent=2), encoding="utf-8")
            upsert_repo_record(snapshot_record, set_active=True)
            _state["repos"] = load_repo_registry().get("repos", [])

            _state["orchestrator"] = Orchestrator(
                vector_index=index,
                graph_agent=graph,
                repo_path=clone_dir,
                history_path=paths["memory_path"],
                repo_name=repo_name,
            )
            _state["indexed"] = True
            _state["file_count"] = len(result.files)
            _state["chunk_count"] = len(result.chunks)
            _state["graph_stats"] = result.graph_stats

            yield _emit(
                "done",
                {
                    "message": (
                        f"Ready! Indexed {len(result.files)} files into {len(result.chunks)} chunks. "
                        f"Changed: {len(result.changed_files)}, reused: {len(result.reused_files)}."
                    ),
                    "progress": 100,
                    "repo_id": repo_id,
                    "repo_name": repo_name,
                    "file_count": len(result.files),
                    "chunk_count": len(result.chunks),
                    "graph_stats": result.graph_stats,
                    "files": [f.rel_path for f in result.files],
                    "changed_files": result.changed_files,
                    "reused_files": result.reused_files,
                    "repos": _state["repos"],
                },
            )
        except Exception as exc:
            logger.exception("Ingestion error")
            yield _emit("error", {"message": str(exc)})

    return EventSourceResponse(generate())


@app.post("/api/query")
async def query_stream(req: QueryRequest):
    async def generate() -> AsyncGenerator:
        try:
            orch = _get_active_orchestrator()
        except HTTPException:
            yield _emit("error", {"message": "No repo indexed yet. Paste a GitHub URL first."})
            return

        yield _emit("thinking", {"agent": "planner", "message": "Planning execution steps..."})

        try:
            loop = asyncio.get_running_loop()
            if req.openai_api_key:
                logger.info("Using user-supplied OpenAI API key for request")
                config.OPENAI_API_KEY = req.openai_api_key
            prepared = await loop.run_in_executor(None, lambda: orch.prepare_query(req.query, focus_file=req.focus_file, api_key=req.openai_api_key))

            for agent in prepared.plan:
                yield _emit("agent", {"agent": agent, "message": f"Ran {agent} agent"})

            if prepared.context.retrieved_files:
                yield _emit("context", {"files": list(dict.fromkeys(prepared.context.retrieved_files))[:8]})

            yield _emit("answer_start", {"citations": prepared.context.citations})
            delta_queue: asyncio.Queue[str | None] = asyncio.Queue()
            stream_task = asyncio.create_task(_stream_reasoning_answer(orch, prepared, delta_queue))

            streamed_parts: list[str] = []
            while True:
                if stream_task.done() and delta_queue.empty():
                    break
                try:
                    delta = await asyncio.wait_for(delta_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                streamed_parts.append(delta)
                yield _emit("answer_delta", {"delta": delta})

            answer_text = await stream_task
            reasoning_ms = max(0.0, response_time_ms(prepared.started_at) - prepared.prep_ms)
            response = await loop.run_in_executor(None, lambda: orch.complete_query(prepared, answer_text, reasoning_ms))

            eval_data = None
            if response.evaluation:
                eval_data = {
                    "overall": response.evaluation.overall_score,
                    "passed": response.evaluation.passed,
                    "issues": response.evaluation.issues,
                }

            yield _emit(
                "answer",
                {
                    "text": response.answer,
                    "plan": response.plan,
                    "metrics": response.metrics,
                    "evaluation": eval_data,
                    "retrieved_files": response.retrieved_files,
                    "repo_name": response.repo_name,
                    "citations": response.citations,
                },
            )
        except Exception as exc:
            logger.exception("Query error")
            yield _emit("error", {"message": str(exc)})

    async def wrapped() -> AsyncGenerator:
        async for event in generate():
            yield event

    return EventSourceResponse(wrapped())


@app.get("/api/status")
async def status():
    if not _state["indexed"]:
        registry = _registry_payload()
        active_repo_id = registry.get("active_repo_id")
        if active_repo_id:
            record = get_repo_record(active_repo_id)
            if record:
                try:
                    _activate_repo_record(record)
                except HTTPException:
                    pass
    registry = _registry_payload()
    return {
        "indexed": _state["indexed"],
        "repo_id": _state["repo_id"],
        "repo_url": _state["repo_url"],
        "repo_name": _state["repo_name"],
        "file_count": _state["file_count"],
        "chunk_count": _state["chunk_count"],
        "graph_stats": _state["graph_stats"],
        "model": config.LLM_MODEL,
        "repos": registry.get("repos", []),
        "active_repo_id": registry.get("active_repo_id", ""),
    }


@app.get("/api/files")
async def list_files():
    orch = _get_active_orchestrator()
    return orch.tools.list_files()


@app.get("/api/history")
async def history(limit: int = 30):
    try:
        orch = _get_active_orchestrator()
    except HTTPException:
        return []
    entries = orch.memory.all_entries()[-limit:]
    return [
        {
            "query": entry.query,
            "answer_preview": entry.answer[:300],
            "score": entry.evaluation_score,
            "timestamp": entry.timestamp,
        }
        for entry in reversed(entries)
    ]


@app.delete("/api/history")
async def clear_history():
    try:
        orch = _get_active_orchestrator()
        orch.memory.clear()
    except HTTPException:
        pass
    return {"status": "cleared"}


@app.post("/api/search")
async def search(body: SearchRequest):
    orch = _get_active_orchestrator()
    extensions = [body.extension] if body.extension else None
    return orch.tools.search_repo(
        body.keyword,
        extensions=extensions,
        case_sensitive=body.case_sensitive,
        regex=body.regex,
        max_results=body.max_results,
    )


@app.get("/api/graph")
async def graph_data():
    try:
        orch = _get_active_orchestrator()
        payload = orch.graph.to_visualization()
        payload["indexed"] = True
        return payload
    except HTTPException:
        return {
            "indexed": False,
            "nodes": [],
            "links": [],
            "stats": {"nodes": 0, "edges": 0, "has_cycles": False, "isolated_files": 0},
            "message": "Index or activate a repo first to see its dependency graph.",
        }


@app.get("/api/repos")
async def list_repos():
    registry = _registry_payload()
    return {
        "active_repo_id": registry.get("active_repo_id", ""),
        "repos": [_repo_summary(r) for r in registry.get("repos", [])],
    }


@app.post("/api/repos/{repo_id}/activate")
async def activate_repo(repo_id: str):
    record = get_repo_record(repo_id)
    if not record:
        raise HTTPException(404, "Repo not found")
    set_active_repo(repo_id)
    orch = _activate_repo_record(record)
    return {
        "status": "ok",
        "active_repo_id": repo_id,
        "repo": _repo_summary(record),
        "files": orch.tools.list_files(),
    }


@app.post("/api/diff-analyze")
async def diff_analyze(body: DiffRequest):
    orch = _get_active_orchestrator()
    resp = await asyncio.get_running_loop().run_in_executor(None, lambda: orch.analyze_diff(body.diff, body.query))
    return {
        "text": resp.answer,
        "plan": resp.plan,
        "metrics": resp.metrics,
        "evaluation": {
            "overall": resp.evaluation.overall_score,
            "passed": resp.evaluation.passed,
            "issues": resp.evaluation.issues,
        } if resp.evaluation else None,
        "retrieved_files": resp.retrieved_files,
    }


@app.get("/api/export/pdf")
async def export_pdf():
    orch = _get_active_orchestrator()
    entries = orch.memory.all_entries()
    lines = [
        f"Repo: {_state.get('repo_name') or orch.repo_name}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for idx, entry in enumerate(entries, 1):
        lines.extend(
            [
                f"Q{idx}: {entry.query}",
                f"A{idx}: {entry.answer[:500]}",
                f"Files: {', '.join(entry.retrieved_files[:8])}",
                "",
            ]
        )
    pdf_bytes = _build_simple_pdf(f"ACIP Session Report - {orch.repo_name}", lines)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{orch.repo_name}-session-report.pdf"'},
    )


@app.get("/", response_class=HTMLResponse)
async def frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Run build step.</h1>", status_code=500)


from fastapi.staticfiles import StaticFiles as _StaticFiles

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", _StaticFiles(directory=str(_static_dir)), name="static")
