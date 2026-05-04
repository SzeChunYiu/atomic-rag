"""Multi-vector (token-level) embedding store + ColBERT-style MaxSim scoring.

We store one matrix per chunk (T_c × d), and at query time aggregate
per-token similarities via MaxSim:

    s(q, c) = Σ_{t in q} max_{u in c}  ⟨q_t, c_u⟩

This is the late-interaction operator from ColBERT (Khattab & Zaharia 2020).
We obtain per-token vectors via the underlying transformer's last-hidden
states + L2 norm — this works with BGE-M3 (its colbert_vecs head exposes
exactly these), with bge-base/bge-large, and with E5 family models.

For the canonical ColBERT-v2 / PLAID baseline used in benchmarks, we leave
a hook to plug in `pylate` later; the in-house implementation here is the
right operator and is sufficient for matched-compute ablations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class MultiVecEncoder(Protocol):
    dim: int

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]: ...


@dataclass
class MultiVecChunk:
    chunk_id: str
    tokens: np.ndarray  # (T, d) float32, L2-normalized rows


class MultiVecIndex:
    def __init__(self, chunks: list[MultiVecChunk]) -> None:
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]

    @classmethod
    def from_texts(
        cls,
        chunk_ids: list[str],
        texts: list[str],
        encoder: MultiVecEncoder,
    ) -> "MultiVecIndex":
        if len(chunk_ids) != len(texts):
            msg = "chunk_ids and texts length mismatch"
            raise ValueError(msg)
        token_mats = encoder.encode_tokens(texts)
        chunks: list[MultiVecChunk] = []
        for cid, mat in zip(chunk_ids, token_mats, strict=True):
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim != 2:
                msg = f"token matrix for {cid} must be 2-D"
                raise ValueError(msg)
            chunks.append(MultiVecChunk(chunk_id=cid, tokens=mat))
        return cls(chunks)

    def maxsim_scores(
        self,
        query_tokens: np.ndarray,
    ) -> dict[str, float]:
        """Return chunk_id → ColBERT MaxSim score."""
        if query_tokens.size == 0:
            return {}
        out: dict[str, float] = {}
        for c in self.chunks:
            sim = c.tokens @ query_tokens.T   # (T_c, T_q)
            per_q = np.max(sim, axis=0)       # (T_q,)
            out[c.chunk_id] = float(np.sum(per_q))
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta_rows: list[dict] = []
        offsets: list[int] = []
        big: list[np.ndarray] = []
        cursor = 0
        for c in self.chunks:
            offsets.append(cursor)
            big.append(c.tokens)
            meta_rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "n_tokens": int(c.tokens.shape[0]),
                    "offset": int(cursor),
                }
            )
            cursor += int(c.tokens.shape[0])
        if big:
            stacked = np.concatenate(big, axis=0)
        else:
            stacked = np.zeros((0, 0), dtype=np.float32)
        np.save(directory / "multivec_tokens.npy", stacked.astype(np.float32))
        (directory / "multivec_meta.jsonl").write_text(
            "\n".join(json.dumps(r) for r in meta_rows) + ("\n" if meta_rows else ""),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "MultiVecIndex":
        stacked = np.load(directory / "multivec_tokens.npy")
        rows = (directory / "multivec_meta.jsonl").read_text(encoding="utf-8").splitlines()
        chunks: list[MultiVecChunk] = []
        for line in rows:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n = int(r["n_tokens"])
            off = int(r["offset"])
            chunks.append(
                MultiVecChunk(
                    chunk_id=str(r["chunk_id"]),
                    tokens=stacked[off : off + n].astype(np.float32),
                )
            )
        return cls(chunks)
