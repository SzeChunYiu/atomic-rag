"""Pseudo-relevance feedback (PRF / RM3-style) for atom retrieval.

Classical IR method (Rocchio 1971; Lavrenko-Croft 2001 RM3) adapted to
sentence-level atoms. After initial top-K retrieval, expand the query
embedding by adding a weighted average of top-M atom embeddings, then
re-retrieve. Typical gain in IR: +2-5pp recall.

Adaptation for RAG: weighting by claim-type-conditioned similarity so
expansion stays on-topic for the query intent (e.g., a WHEN query
expands toward dates, not arbitrary tokens). This is the RAG-native
twist that turns vanilla PRF into something tuned for atomic retrieval.
"""
from __future__ import annotations
import numpy as np


def prf_expand_query(
    query_emb: np.ndarray,        # (D,)
    atom_embs: np.ndarray,        # (N, D), L2-normalized rows
    atom_types: np.ndarray,       # (N,) of str
    top_m: int = 10,
    alpha: float = 0.7,           # weight of original query
    intent_type: str = "ANY",
    intent_weight: float = 0.2,   # extra weight for atoms matching intent
) -> np.ndarray:
    """Return expanded query embedding.

    Original PRF: q' = alpha*q + (1-alpha) * mean(top_M atom embs).
    Adapted: when intent is type-specific, weight type-matching atoms
    by an extra factor to keep expansion on-topic.
    """
    s = atom_embs @ query_emb
    if intent_type != "ANY" and intent_weight > 0:
        s = s + intent_weight * (atom_types == intent_type).astype(np.float32)
    top_idx = np.argpartition(-s, min(top_m, len(s) - 1))[:top_m]
    feedback = atom_embs[top_idx].mean(axis=0)
    feedback = feedback / (np.linalg.norm(feedback) + 1e-9)
    expanded = alpha * query_emb + (1 - alpha) * feedback
    return expanded / (np.linalg.norm(expanded) + 1e-9)
