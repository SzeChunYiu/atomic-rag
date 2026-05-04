"""Persist chunk inventories alongside dense/BM25 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from astro_cs_rag.atoms.schemas import Chunk


def save_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [c.model_dump_json() for c in chunks]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    rows: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Chunk.model_validate_json(line))
    return rows


def save_index_meta(directory: Path, meta: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_index_meta(directory: Path) -> dict[str, object]:
    raw = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
    return dict(raw)
