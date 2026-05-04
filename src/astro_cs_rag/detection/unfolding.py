"""Score-based unfolding of embedding smearing.

Conceptual model: the observed retrieval score field y(i) is a noisy version
of a latent relevance distribution x(i). We assume

    y = x + sigma * eta,    eta ~ N(0, I)

with sigma estimated from the score-shape (calorimetry). The 'diffusion
prior' is replaced here by a soft-thresholded Gaussian prior p(x) ∝ exp(-x.T H x),
where H is a graph-Laplacian regularizer encouraging smoothness over the
chunk-similarity graph. The MAP solution is

    x_hat = argmin_x  || y - x ||^2 / (2 sigma^2)  +  lambda * x.T L x

which reduces to a linear solve x_hat = (I + 2 lambda sigma^2 L)^{-1} y.

For full diffusion-prior unfolding (Howard et al. 2024), this minimal
solver is replaced by a learned reverse SDE; structurally the interface is
the same: in → noisy y; out → denoised x.
"""

from __future__ import annotations

import numpy as np


def graph_laplacian(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    W = np.zeros_like(sim)
    for i in range(n):
        nbrs = np.argsort(-sim[i])[:k]
        for j in nbrs:
            W[i, j] = max(0.0, float(sim[i, j]))
    W = (W + W.T) / 2.0
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L.astype(np.float32)


def unfold_scores(
    chunk_ids: list[str],
    raw_scores: dict[str, float],
    embeddings_by_id: dict[str, np.ndarray],
    *,
    sigma: float = 0.5,
    laplacian_lambda: float = 0.5,
    k_nn: int = 5,
) -> dict[str, float]:
    if not chunk_ids:
        return {}
    embs = []
    cids: list[str] = []
    y_vals = []
    for c in chunk_ids:
        v = embeddings_by_id.get(c)
        if v is None:
            continue
        embs.append(v)
        cids.append(c)
        y_vals.append(float(raw_scores.get(c, 0.0)))
    if not cids:
        return {c: float(raw_scores.get(c, 0.0)) for c in chunk_ids}
    E = np.stack(embs, axis=0).astype(np.float32)
    L = graph_laplacian(E, k=min(k_nn, len(cids) - 1))
    A = np.eye(len(cids), dtype=np.float32) + 2.0 * float(laplacian_lambda) * (sigma ** 2) * L
    y = np.asarray(y_vals, dtype=np.float32)
    x = np.linalg.solve(A, y)
    return {cids[i]: float(x[i]) for i in range(len(cids))}
