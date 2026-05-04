"""Coronagraphy: mask the dominant anchor chunk and re-rank the residual.

In coronagraphy a small disk occults the bright on-axis source so faint
companions become visible. Translated:

  1. Run normal retrieval and identify the top-1 (or top-K_anchor) candidates.
  2. Subtract the *projection* of each remaining candidate onto each anchor
     (in score space, with embedding-similarity weights) — this is the residual
     score field.
  3. Rank candidates by *residual* relevance.

The output is a complementary candidate ordering that surfaces evidence the
top-1 was drowning out — typical for F5 (popular-but-empty) and F4 (split
evidence).
"""

from __future__ import annotations

import numpy as np


def coronagraph_residual(
    *,
    chunk_ids: list[str],
    scores: dict[str, float],
    embeddings_by_id: dict[str, np.ndarray],
    n_anchors: int = 1,
) -> dict[str, float]:
    if not chunk_ids:
        return {}
    ranked = sorted(chunk_ids, key=lambda cid: -scores.get(cid, 0.0))
    anchors = ranked[:n_anchors]
    if not anchors:
        return dict(scores)

    anchor_emb = []
    for a in anchors:
        v = embeddings_by_id.get(a)
        if v is None:
            continue
        v = v / (np.linalg.norm(v) + 1e-9)
        anchor_emb.append(v)
    if not anchor_emb:
        return dict(scores)
    A = np.stack(anchor_emb, axis=0).astype(np.float32)

    residual: dict[str, float] = {}
    for cid in chunk_ids:
        if cid in anchors:
            residual[cid] = 0.0  # anchors are masked
            continue
        v = embeddings_by_id.get(cid)
        if v is None:
            residual[cid] = float(scores.get(cid, 0.0))
            continue
        v = v / (np.linalg.norm(v) + 1e-9)
        sim = A @ v.astype(np.float32)            # (n_anchors,)
        weight = float(np.max(np.maximum(sim, 0.0)))
        residual[cid] = float(scores.get(cid, 0.0)) * (1.0 - weight)
    return residual
