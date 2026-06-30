"""
ingestion/pipeline.py - Shared ingestion pipeline with incremental re-indexing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agents.graph_agent import GraphAgent
from ingestion.chunker import chunk_codebase
from ingestion.indexer import VectorIndex
from ingestion.loader import CodeFile, RepoLoader
from ingestion.manifest import IngestManifest, IngestedFile
from ingestion.parser import parse_source_file
from ingestion.summarizer import summarize_files_async

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, int, dict], Awaitable[None] | None]


@dataclass
class IngestionResult:
    repo_path: Path
    files: list[CodeFile]
    parsed: dict
    summaries: dict[str, str]
    chunks: list
    graph_stats: dict
    changed_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    reused_files: list[str] = field(default_factory=list)
    durations_ms: dict[str, float] = field(default_factory=dict)


async def ingest_repository(
    repo_path: Path,
    index: VectorIndex,
    graph: GraphAgent,
    manifest_path: Path,
    progress: ProgressCallback | None = None,
) -> IngestionResult:
    manifest = IngestManifest.load(manifest_path)
    index.load()

    durations_ms: dict[str, float] = {}

    async def _progress(phase: str, message: str, pct: int, **extra) -> None:
        if progress is None:
            return
        maybe_awaitable = progress(phase, message, pct, extra)
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

    t0 = time.perf_counter()
    await _progress("load", "Scanning source files...", 20)
    files = await asyncio.to_thread(lambda: RepoLoader(root_dir=repo_path).load())
    durations_ms["load"] = (time.perf_counter() - t0) * 1000

    current_hashes = {cf.rel_path: cf.metadata.get("sha256", "") for cf in files}
    previous_hashes = {rel_path: record.sha256 for rel_path, record in manifest.files.items()}
    changed_files = sorted(
        rel_path for rel_path, sha256 in current_hashes.items()
        if previous_hashes.get(rel_path) != sha256
    )
    removed_files = sorted(rel_path for rel_path in previous_hashes if rel_path not in current_hashes)
    reused_files = sorted(rel_path for rel_path in current_hashes if rel_path not in changed_files)

    t0 = time.perf_counter()
    await _progress("parse", f"Parsing {len(files)} source files...", 30, changed_files=changed_files)
    parsed = {
        cf.rel_path: await asyncio.to_thread(parse_source_file, cf.content, cf.rel_path)
        for cf in files
    }
    durations_ms["parse"] = (time.perf_counter() - t0) * 1000

    changed_file_objects = [cf for cf in files if cf.rel_path in changed_files]
    summaries = {rel_path: record.summary for rel_path, record in manifest.files.items() if rel_path in reused_files}

    async def _on_summary_progress(done: int, total: int, rel_path: str) -> None:
        pct = 45 if total == 0 else 45 + int((done / total) * 20)
        await _progress(
            "summarize",
            f"Summarised {done}/{total}: {rel_path}",
            pct,
            changed_files=changed_files,
            reused_count=len(reused_files),
        )

    t0 = time.perf_counter()
    await _progress(
        "summarize",
        f"Summarising {len(changed_file_objects)} changed files ({len(reused_files)} reused)...",
        45,
        changed_files=changed_files,
        reused_count=len(reused_files),
    )
    summaries = await summarize_files_async(
        changed_file_objects,
        parsed_modules=parsed,
        existing_summaries=summaries,
        on_progress=_on_summary_progress if changed_file_objects else None,
    )
    durations_ms["summarize"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    await _progress("chunk", "Chunking codebase...", 68)
    chunks = await asyncio.to_thread(chunk_codebase, files, parsed, summaries)
    durations_ms["chunk"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    await _progress("index", "Updating vector index...", 78)
    await asyncio.to_thread(index.rebuild_incremental, chunks, set(changed_files))
    await asyncio.to_thread(index.save)
    durations_ms["index"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    await _progress("graph", "Building dependency graph...", 90)
    await asyncio.to_thread(graph.build, parsed)
    await asyncio.to_thread(graph.save)
    graph_stats = graph.summary_stats()
    durations_ms["graph"] = (time.perf_counter() - t0) * 1000

    chunks_by_file: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_file[chunk.file_path] = chunks_by_file.get(chunk.file_path, 0) + 1

    next_manifest = IngestManifest(repo_path=str(repo_path), generated_at=time.time())
    next_manifest.files = {
        cf.rel_path: IngestedFile(
            rel_path=cf.rel_path,
            sha256=cf.metadata.get("sha256", ""),
            summary=summaries.get(cf.rel_path, ""),
            size_bytes=cf.size_bytes,
            lines=cf.lines,
            chunk_count=chunks_by_file.get(cf.rel_path, 0),
        )
        for cf in files
    }
    await asyncio.to_thread(next_manifest.save, manifest_path)

    return IngestionResult(
        repo_path=repo_path,
        files=files,
        parsed=parsed,
        summaries=summaries,
        chunks=chunks,
        graph_stats=graph_stats,
        changed_files=changed_files,
        removed_files=removed_files,
        reused_files=reused_files,
        durations_ms=durations_ms,
    )
