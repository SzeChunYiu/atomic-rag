"""Sinkhorn-OT retrieval: query and chunk as token-measures, score by 1 - W2.

We treat a query and a chunk as discrete distributions over their token
embeddings (uniform weights). The negative entropy-regularized OT distance
$-\\mathrm{OT}_\\epsilon$ is the score; chunks with low OT cost to the query
are ranked higher.

This module implements pure-Python Sinkhorn iterations sufficient for
benchmark candidate-pool sizes; for large N use POT (`pip install pot`).
"""

from __future__ import annotations

import numpy as np


def cost_matrix(a_emb: np.ndarray, b_emb: np.ndarray) -> np.ndarray:
    """Cosine-distance cost matrix in [0, 2]."""
    a = a_emb / (np.linalg.norm(a_emb, axis=1, keepdims=True) + 1e-9)
    b = b_emb / (np.linalg.norm(b_emb, axis=1, keepdims=True) + 1e-9)
    sim = a @ b.T
    return np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)


def sinkhorn(
    cost: np.ndarray,
    a_weights: np.ndarray,
    b_weights: np.ndarray,
    *,
    epsilon: float = 0.1,
    n_iters: int = 50,
) -> tuple[np.ndarray, float]:
    """Entropy-regularized OT via the Sinkhorn–Knopp iteration.

    Returns (transport_plan, OT_cost). All inputs must be non-negative; weight
    vectors must sum to 1 (within tolerance).
    """
    K = np.exp(-cost / max(epsilon, 1e-9)).astype(np.float64)
    a = a_weights.astype(np.float64) + 1e-12
    b = b_weights.astype(np.float64) + 1e-12
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = u[:, None] * K * v[None, :]
    ot = float(np.sum(P * cost))
    return P, ot


def wasserstein_score(
    query_token_emb: np.ndarray,
    chunk_token_emb: np.ndarray,
    *,
    epsilon: float = 0.1,
    n_iters: int = 30,
) -> float:
    """Negative OT distance — higher = closer."""
    if query_token_emb.size == 0 or chunk_token_emb.size == 0:
        return 0.0
    cost = cost_matrix(query_token_emb, chunk_token_emb)
    a = np.ones(cost.shape[0], dtype=np.float64) / cost.shape[0]
    b = np.ones(cost.shape[1], dtype=np.float64) / cost.shape[1]
    _, ot = sinkhorn(cost, a, b, epsilon=epsilon, n_iters=n_iters)
    return float(-ot)
