"""Token-level encoder backed by sentence-transformers (no extra dep)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class MultiVecEncoder(Protocol):
    dim: int

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]: ...


class STMultiVecEncoder:
    """Per-token L2-normalized embeddings via sentence-transformers transformer module.

    Works for any HF model with a transformer backbone (BGE-M3, BGE-large, E5,
    GTE). For ColBERT-v2-PLAID-grade scoring, plug in `pylate` later via the
    `external_pylate` constructor — the index format is compatible.
    """

    dim: int

    def __init__(self, model_name: str, max_seq_length: int = 256) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = int(max_seq_length)
        self._tokenizer = self._model.tokenizer
        # probe dim
        with self._model._target_device:  # type: ignore[attr-defined]
            pass
        sample = self._encode_one("dim probe")
        self.dim = int(sample.shape[1])

    def _encode_one(self, text: str) -> np.ndarray:
        import torch

        tokens = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self._model.max_seq_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        first = self._model._first_module()  # transformer
        with torch.no_grad():
            out = first.auto_model(**tokens, output_hidden_states=False)
        last = out.last_hidden_state[0]                  # (T, d)
        mask = tokens["attention_mask"][0].bool()
        last = last[mask]
        last = torch.nn.functional.normalize(last, dim=-1)
        return last.cpu().to(torch.float32).numpy()

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for t in texts:
            out.append(self._encode_one(t))
        return out


class HashMultiVecEncoder:
    """Deterministic CI-friendly token encoder — token = whitespace word."""

    dim: int = 32

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        import hashlib

        out: list[np.ndarray] = []
        for t in texts:
            words = t.split() or [t]
            rows: list[np.ndarray] = []
            for w in words:
                h = hashlib.sha256(w.lower().encode("utf-8")).digest()
                buf = (h * ((self.dim // len(h)) + 1))[: self.dim]
                v = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                v = v / (float(np.linalg.norm(v)) + 1e-9)
                rows.append(v)
            out.append(np.stack(rows, axis=0))
        return out
