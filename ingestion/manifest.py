"""
ingestion/manifest.py - Persisted metadata for incremental re-ingestion.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class IngestedFile:
    rel_path: str
    sha256: str
    summary: str = ""
    size_bytes: int = 0
    lines: int = 0
    chunk_count: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass
class IngestManifest:
    repo_path: str = ""
    generated_at: float = field(default_factory=time.time)
    files: dict[str, IngestedFile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IngestManifest":
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()

        files = {
            rel_path: IngestedFile(**data)
            for rel_path, data in payload.get("files", {}).items()
        }
        return cls(
            repo_path=payload.get("repo_path", ""),
            generated_at=payload.get("generated_at", 0.0),
            files=files,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_path": self.repo_path,
            "generated_at": self.generated_at,
            "files": {rel_path: asdict(record) for rel_path, record in self.files.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
