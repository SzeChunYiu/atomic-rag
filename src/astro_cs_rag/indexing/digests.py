"""Compute reproducibility-grade digests of embedder / reranker / generator backends."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def model_revision_digest(model_name: str, *, max_files: int = 32) -> dict[str, Any]:
    """Best-effort fingerprint of a sentence-transformers model directory.

    We hash the file names + sizes of files in the HF cache snapshot for
    `model_name`, if a snapshot can be located. If not (e.g. CI without the
    cache), returns a sentinel without raising — this avoids gating dev work
    on full model downloads but keeps the field present in artifacts.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        local_dir = Path(snapshot_download(model_name, allow_patterns=["*.json", "*.txt"]))
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"model_name": model_name, "revision": None, "error": str(exc)}

    h = hashlib.sha256()
    file_listing: list[tuple[str, int]] = []
    for p in sorted(local_dir.rglob("*"))[:max_files]:
        if p.is_file():
            file_listing.append((p.name, p.stat().st_size))
            h.update(p.name.encode("utf-8"))
            h.update(str(p.stat().st_size).encode("utf-8"))
    return {
        "model_name": model_name,
        "revision": h.hexdigest()[:16],
        "files_sampled": file_listing,
    }
