"""Embedding backends — swap models without touching retrieval math."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceEmbedder:
    dim: int

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch = batch_size
        test = self._model.encode(["probe"], batch_size=1)
        self.dim = int(test.shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            texts,
            batch_size=self._batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)


class HashEmbedder:
    """Deterministic low-dimensional vectors for fast CI tests (no torch)."""

    dim = 32

    def encode(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            buf = (digest * ((self.dim // len(digest)) + 1))[: self.dim]
            vec = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            vec = vec / (float(np.linalg.norm(vec)) + 1e-9)
            rows.append(vec)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows, axis=0)
