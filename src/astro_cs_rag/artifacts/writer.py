"""Centralized run outputs — modules should not write run files ad hoc."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel


class ArtifactWriter:
    def __init__(self, root: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.root = root / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def attach(cls, run_dir: Path) -> ArtifactWriter:
        """Write into an existing run directory (extend retrieve → detect → …)."""
        inst = cls.__new__(cls)
        resolved = run_dir.resolve()
        inst.root = resolved
        inst.run_id = resolved.name
        return inst

    @property
    def run_path(self) -> Path:
        return self.root

    def write_config_snapshot(self, config: BaseModel | dict[str, Any]) -> Path:
        path = self.root / "config.yaml"
        if isinstance(config, BaseModel):
            payload = config.model_dump(mode="json")
        else:
            payload = dict(config)
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return path

    def write_manifest(self, manifest: BaseModel) -> Path:
        path = self.root / "manifest.json"
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_jsonl(self, name: str, rows: Iterable[BaseModel | dict[str, Any]]) -> Path:
        path = self.root / name
        lines: list[str] = []
        for row in rows:
            if isinstance(row, BaseModel):
                lines.append(row.model_dump_json())
            else:
                lines.append(json.dumps(row, ensure_ascii=False))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        path = self.root / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_markdown(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def touch_started(self) -> Path:
        meta = {"run_id": self.run_id, "started_at": datetime.now(tz=UTC).isoformat()}
        return self.write_json("run_meta.json", meta)
