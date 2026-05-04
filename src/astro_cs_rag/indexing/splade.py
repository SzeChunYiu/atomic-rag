"""SPLADE-style learned-sparse retrieval.

We use the standard SPLADE recipe:
  1. Forward the text through a masked-LM head.
  2. Take log(1 + ReLU(logits)) for each (position, vocab_id) cell.
  3. Max-pool over positions per vocab_id → a single sparse vector over the
     vocabulary, with most cells exactly zero.

Score is the dot product of two such sparse vectors. At our benchmark scale
(<= 100k chunks) we just dense-stack and dot-product; an inverted-index
implementation can be added later if scale demands it.

A `HashSpladeBackend` is provided as a CI-friendly stub (no transformers
download).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SpladeBackend(Protocol):
    vocab_size: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashSpladeBackend:
    """Deterministic, no-network stub. Hashes whitespace tokens to vocab cells."""

    name: str = "hash_splade_stub"
    vocab_size: int = 4096

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                idx = (hash(tok) & 0xFFFF) % self.vocab_size
                out[i, idx] += 1.0
            out[i] = np.log(1.0 + out[i])
        return out


class HFSpladeBackend:
    """Real SPLADE backend using a HF MLM model — invoked only in non-CI runs.

    Default model: `naver/splade-v3` (Apache-2.0). For lighter compute use
    `naver/splade-cocondenser-ensembledistil`.
    """

    name: str
    vocab_size: int

    def __init__(self, model_name: str = "naver/splade-v3", batch_size: int = 8) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.train(False)
        self.name = model_name
        self.vocab_size = int(self.tokenizer.vocab_size)
        self._batch = int(batch_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        out = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            mask = enc["attention_mask"].unsqueeze(-1)
            relu = torch.relu(logits) * mask
            saturated = torch.log1p(relu)
            pooled, _ = saturated.max(dim=1)
            out[start : start + len(batch)] = pooled.cpu().to(torch.float32).numpy()
        return out


class SpladeIndex:
    def __init__(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        backend_name: str,
    ) -> None:
        if len(chunk_ids) != vectors.shape[0]:
            msg = "chunk_ids and vectors length mismatch"
            raise ValueError(msg)
        self.chunk_ids = list(chunk_ids)
        self.vectors = vectors.astype(np.float32, copy=False)
        self.backend_name = backend_name

    @classmethod
    def from_texts(
        cls,
        chunk_ids: list[str],
        texts: list[str],
        backend: SpladeBackend,
    ) -> "SpladeIndex":
        vecs = backend.encode(texts)
        return cls(chunk_ids=chunk_ids, vectors=vecs, backend_name=backend.name)

    def scores(self, query_vec: np.ndarray) -> dict[str, float]:
        if query_vec.size == 0 or self.vectors.size == 0:
            return {}
        if query_vec.shape[0] != self.vectors.shape[1]:
            msg = "splade dim mismatch"
            raise ValueError(msg)
        sims = self.vectors @ query_vec.astype(np.float32)
        return {self.chunk_ids[i]: float(sims[i]) for i in range(len(self.chunk_ids))}
