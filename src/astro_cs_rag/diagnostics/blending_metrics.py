"""Chunk-vs-atom blending diagnostics.

A bridge atom can have a high `cos(q, atom)` while its carrier chunk has
low `cos(q, chunk)`. Chunk-level retrieval/SNR then suppresses the atom.
We call this gap the *blending gap*; large positive gaps are evidence
that chunk-level scoring is hiding signal.
"""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors. Inputs need not be unit-norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def blending_gap(
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    atom_embs: np.ndarray,
) -> float:
    """`max_a cos(q, a) - cos(q, chunk)` for atoms in one chunk.

    Positive gap → the chunk hides a stronger atom than the chunk score.
    Zero gap → chunk score is already at least as good as any atom.
    Negative gap is possible if chunk pooling boosts above any single
    atom (rare with mean pooling on similar sentences).
    """
    if atom_embs.size == 0:
        return 0.0
    if atom_embs.ndim == 1:
        atom_embs = atom_embs.reshape(1, -1)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    c_score = cosine(query_emb, chunk_emb)
    norms = np.linalg.norm(atom_embs, axis=1) + 1e-12
    a_scores = (atom_embs @ q) / norms
    return float(a_scores.max() - c_score)


def coarsening_loss(
    gold_atom_recall_at_k_atom: float,
    gold_atom_recall_at_k_chunk: float,
) -> float:
    """How much retrieval recall is lost by switching atom → chunk index.

    Defined so positive values mean atom-level retrieval is better, which
    is the regime that motivates the deblending direction. Inputs are
    fractions in [0, 1].
    """
    return float(gold_atom_recall_at_k_atom - gold_atom_recall_at_k_chunk)


def per_chunk_blending_table(
    query_emb: np.ndarray,
    chunk_embs: np.ndarray,
    atom_embs: np.ndarray,
    atom_to_chunk_idx: np.ndarray,
) -> np.ndarray:
    """Vectorised blending gap per chunk. Returns array of shape (n_chunks,).

    `atom_to_chunk_idx[i]` is the chunk index for atom i. Chunks with no
    atoms get a gap of 0.0.
    """
    n_chunks = chunk_embs.shape[0]
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    c_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-12
    chunk_scores = (chunk_embs @ q) / c_norms
    if atom_embs.size == 0:
        return np.zeros(n_chunks, dtype=np.float32)
    a_norms = np.linalg.norm(atom_embs, axis=1) + 1e-12
    atom_scores = (atom_embs @ q) / a_norms
    best_atom = np.full(n_chunks, -np.inf, dtype=np.float64)
    np.maximum.at(best_atom, atom_to_chunk_idx, atom_scores)
    seen = np.zeros(n_chunks, dtype=bool)
    seen[atom_to_chunk_idx] = True
    gaps = np.where(seen, best_atom - chunk_scores, 0.0)
    return gaps.astype(np.float32)
