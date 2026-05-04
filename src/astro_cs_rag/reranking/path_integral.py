"""Path-Integral Retrieval (PIR) — rerank by multi-hop chain support.

Idea: a chunk's relevance to a multi-hop query is not just its direct
embedding similarity cos(q, v_i), but also its cumulative *path-integral*
support — the weighted sum over paths from query-similar source chunks
through intermediate chunks to the candidate.

Mathematically, with chunk-chunk similarity matrix A and query-chunk
score vector s, the path-integral score of chunk j is:

    score[j] = s[j]                           (direct, length 0)
             + sum_{k=1..K} sum_i s[i] * (A^k)[i,j] / k

where A is the cosine similarity above threshold tau, and the 1/k
normalization down-weights long paths (heuristic; can be replaced by
a temperature beta).

This captures bridging chains that single-hop scoring cannot:
- Chunk A (high s, mentions intermediate entity Y)
- Chunk B (low s, but mentions Y and answer Z)
- A-B edge (moderate cos via Y), so B receives propagated support from A.

Output: reordered top-N chunk_ids sorted by path-integral score.
"""

from __future__ import annotations

import numpy as np


def path_integral_rerank(
    query_emb: np.ndarray,
    chunk_embs: np.ndarray,
    chunk_ids: list[str],
    *,
    edge_threshold: float = 0.5,
    max_path_length: int = 3,
    direct_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Return chunks reordered by path-integral score (descending).

    Args:
        query_emb: (d,) — pre-normalized query embedding.
        chunk_embs: (N, d) — pre-normalized chunk embeddings.
        chunk_ids: length-N list of chunk_id strings.
        edge_threshold: cos similarity above this counts as a graph edge.
        max_path_length: K — paths of length 1..K contribute.
        direct_weight: weight for direct query-chunk score.
    """
    if len(chunk_ids) == 0:
        return []
    s = chunk_embs @ query_emb  # (N,) direct query similarity
    sim = chunk_embs @ chunk_embs.T  # (N, N) chunk-chunk similarity
    np.fill_diagonal(sim, 0.0)
    # Threshold: only edges above tau, weighted by similarity
    A = np.where(sim > edge_threshold, sim, 0.0).astype(np.float32)

    score = direct_weight * s.astype(np.float32)
    M = A.copy()
    for k in range(1, max_path_length + 1):
        # contribution[j] = sum_i s[i] * (A^k)[i, j] / k
        score += (s @ M) / float(k)
        if k < max_path_length:
            M = M @ A

    order = np.argsort(-score)
    return [(chunk_ids[int(i)], float(score[int(i)])) for i in order]
