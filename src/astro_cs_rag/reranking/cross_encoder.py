"""Cross-encoder reranker (bge-reranker-v2-m3) for the dense+rerank baseline."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    name: str

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        self._model = CrossEncoder(model_name)
        self._batch = batch_size
        self.name = model_name

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(pairs, batch_size=self._batch)
        return np.asarray(raw, dtype=np.float32)


class IdentityReranker:
    """Score = original retrieval score; used as a CI-friendly fallback."""

    name = "identity"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.zeros((len(passages),), dtype=np.float32)


def softmax_score_norm(raw: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if raw.size == 0:
        return raw
    z = raw / max(temperature, 1e-9)
    z = z - float(np.max(z))
    exps = np.exp(z)
    s = float(np.sum(exps))
    if s <= 0 or math.isnan(s):
        return np.zeros_like(raw)
    return exps / s
