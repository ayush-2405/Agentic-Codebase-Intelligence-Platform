from __future__ import annotations

import json
import re
import time
from pathlib import Path

import config


def _read_registry() -> dict:
    if not config.REPO_REGISTRY_PATH.exists():
        return {"active_repo_id": "", "repos": []}
    try:
        return json.loads(config.REPO_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"active_repo_id": "", "repos": []}


def _write_registry(data: dict) -> None:
    config.REPO_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REPO_REGISTRY_PATH.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def slugify_repo(repo_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", repo_name.strip().lower()).strip("-")
    return slug or "repo"


def make_repo_id(repo_name: str) -> str:
    return f"{slugify_repo(repo_name)}-{int(time.time())}"


def repo_snapshot_dir(repo_id: str) -> Path:
    return config.REPO_STORE_DIR / repo_id


def repo_record_paths(repo_id: str) -> dict:
    base = repo_snapshot_dir(repo_id)
    return {
        "base": base,
        "index_path": base / "faiss.index",
        "chunks_path": base / "chunks.pkl",
        "vectors_path": base / "vectors.npy",
        "graph_path": base / "dependency_graph.pkl",
        "memory_path": base / "chat_history.json",
        "meta_path": base / "meta.json",
        "manifest_path": base / "manifest.json",
    }


def upsert_repo_record(record: dict, set_active: bool = True) -> dict:
    registry = _read_registry()
    repos = [r for r in registry.get("repos", []) if r.get("repo_id") != record.get("repo_id")]
    repos.append(record)
    repos.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
    registry["repos"] = repos
    if set_active:
        registry["active_repo_id"] = record.get("repo_id", "")
    _write_registry(registry)
    return registry


def list_repos() -> dict:
    return _read_registry()


def set_active_repo(repo_id: str) -> dict:
    registry = _read_registry()
    registry["active_repo_id"] = repo_id
    _write_registry(registry)
    return registry


def get_repo_record(repo_id: str) -> dict | None:
    registry = _read_registry()
    for record in registry.get("repos", []):
        if record.get("repo_id") == repo_id:
            return record
    return None


def find_repo_record(repo_url: str, branch: str = "") -> dict | None:
    registry = _read_registry()
    for record in registry.get("repos", []):
        if record.get("repo_url") == repo_url and record.get("branch", "") == branch:
            return record
    return None
