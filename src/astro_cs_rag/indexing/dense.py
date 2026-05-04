"""Dense cosine retrieval over precomputed embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunk_ids) != embeddings.shape[0]:
            msg = "chunk_ids and embeddings row count mismatch"
            raise ValueError(msg)
        self.chunk_ids = chunk_ids
        self.embeddings = embeddings.astype(np.float32, copy=False)
        # Pre-normalize once at construction. Previously scores() renormalized
        # the entire matrix on every query — quadratic waste over many queries.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self._normed = (self.embeddings / norms).astype(np.float32, copy=False)

    def scores(self, query_embedding: np.ndarray) -> dict[str, float]:
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            msg = "embedding dimension mismatch"
            raise ValueError(msg)
        q = query_embedding.astype(np.float32, copy=False)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._normed @ q
        # Return as dict (some callers iterate over chunk_ids); building the
        # dict is unavoidable while preserving the existing interface, but
        # the matmul is now O(N×D) once instead of O(N×D) renorm + matmul.
        return dict(zip(self.chunk_ids, sims.tolist(), strict=True))

    def topk(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) without building a full dict."""
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            msg = "embedding dimension mismatch"
            raise ValueError(msg)
        q = query_embedding.astype(np.float32, copy=False)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._normed @ q
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part])]
        return [(self.chunk_ids[int(i)], float(sims[int(i)])) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        meta = {"chunk_ids": self.chunk_ids, "dim": int(self.embeddings.shape[1])}
        (directory / "dense_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> DenseIndex:
        emb = np.load(directory / "embeddings.npy")
        meta = json.loads((directory / "dense_meta.json").read_text(encoding="utf-8"))
        return cls(list(meta["chunk_ids"]), emb)
