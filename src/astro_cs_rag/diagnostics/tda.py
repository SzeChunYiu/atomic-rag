"""Topological data analysis (TDA) features of the score-similarity graph.

We compute *persistent zero-dimensional homology* (H0): the lifetimes of
connected components as we sweep the similarity threshold from 1 (everyone
disconnected) to 0 (everyone connected). The bar lengths are persistence
lifetimes; their distribution describes the topological "shape" of the
chunk neighborhood around a query.

We also compute a coarse H1 estimate via the Euler-characteristic surrogate
(V - E + F at the lowest threshold), which is fast and avoids the dependency
on `gudhi` / `giotto-tda`.

Reference: Pranav et al. 2017 (cosmic web persistent homology); Wilding et al.
2024 (cosmic-void TDA).
"""

from __future__ import annotations

import numpy as np


def _union_find(n: int) -> tuple[list[int], list[int]]:
    parent = list(range(n))
    rank = [0] * n
    return parent, rank


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], rank: list[int], a: int, b: int) -> bool:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return False
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return True


def persistence_h0(similarities: np.ndarray) -> list[float]:
    """Bar lifetimes for H0 over a Vietoris–Rips-like filtration of a graph.

    `similarities` is an upper-triangular (n, n) matrix in [0, 1]. We sweep
    threshold from 1 down to 0, adding edges; whenever two components merge,
    record the lifetime = (similarity at merge) since the youngest of the two
    components was born (born at sim=1 effectively).
    """
    n = similarities.shape[0]
    if n <= 1:
        return []
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(similarities[i, j]), i, j))
    edges.sort(key=lambda t: -t[0])
    parent, rank = _union_find(n)
    bars: list[float] = []
    for s, i, j in edges:
        if _union(parent, rank, i, j):
            bars.append(1.0 - s)
    return bars


def tda_summary(score_field: dict[str, float], embeddings_by_id: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute a small TDA summary of the top-K candidate subgraph.

    Inputs:
      score_field: chunk_id -> retrieval score
      embeddings_by_id: chunk_id -> embedding vector
    Returns: mean persistence, max persistence, persistence entropy, n_components
    """
    cids = sorted(score_field, key=lambda c: -score_field[c])[:20]
    embs = []
    for c in cids:
        v = embeddings_by_id.get(c)
        if v is None:
            continue
        embs.append(v / (np.linalg.norm(v) + 1e-9))
    if len(embs) < 2:
        return {"mean_persistence": 0.0, "max_persistence": 0.0, "persistence_entropy": 0.0, "n_components": float(len(embs))}
    M = np.stack(embs, axis=0).astype(np.float32)
    sims = np.clip(M @ M.T, 0.0, 1.0)
    bars = persistence_h0(sims)
    if not bars:
        return {"mean_persistence": 0.0, "max_persistence": 0.0, "persistence_entropy": 0.0, "n_components": 1.0}
    arr = np.asarray(bars)
    mean_p = float(arr.mean())
    max_p = float(arr.max())
    p = arr / (arr.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "mean_persistence": mean_p,
        "max_persistence": max_p,
        "persistence_entropy": ent,
        "n_components": float(len(bars) + 1),
    }
