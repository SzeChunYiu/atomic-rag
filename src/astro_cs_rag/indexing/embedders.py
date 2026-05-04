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
    """Deterministic low-dimensional vectors for fast CI tests (no torch).

    NOTE: this is a hash of the *whole string* — it has no token- or
    content-level signal, so cos(query, atom) is essentially random
    across paraphrases. Use :class:`TrigramEmbedder` for benchmarks
    that need lexical signal without a real model.
    """

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


class TrigramEmbedder:
    """Hashed character-trigram TF — CPU-only, deterministic, has signal.

    Each text becomes a `dim`-dim L2-normalised vector of hashed
    character-trigram counts (lowercased). Cosine of two encodings is
    a noisy approximation of trigram-Jaccard, which is enough to make
    cos(query, semantically-related atom) > cos(query, unrelated atom)
    without pulling in a transformer model.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = int(dim)

    def _encode_one(self, text: str) -> np.ndarray:
        s = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        if len(s) < 3:
            s = s + "  "  # pad so we always emit at least one trigram
        for i in range(len(s) - 2):
            tri = s[i : i + 3]
            h = int.from_bytes(hashlib.blake2b(tri.encode("utf-8"), digest_size=4).digest(), "little")
            vec[h % self.dim] += 1.0
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(t) for t in texts], axis=0)
