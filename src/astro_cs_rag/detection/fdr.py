"""Benjamini-Hochberg FDR control on per-query candidate z-scores.

Replaces a fixed top-k cut with a calibrated multiple-testing threshold:
the candidates kept have an expected false-discovery rate <= alpha.
The null hypothesis per candidate is "score drawn from background"; we
convert each z to a one-sided normal p-value, then apply the BH procedure.

This is exactly how astronomy catalog construction handles spurious
detections at scale (Zackay & Ofek 2017; Coronograph residual catalogs).
"""

from __future__ import annotations

import math


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def benjamini_hochberg(
    pvalues: list[float],
    alpha: float = 0.10,
) -> list[bool]:
    """Return a boolean mask: True = reject null (= keep candidate)."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    sorted_p = [pvalues[i] for i in order]
    threshold_rank = -1
    for k, p in enumerate(sorted_p, start=1):
        if p <= alpha * k / n:
            threshold_rank = k
    keep = [False] * n
    if threshold_rank == -1:
        return keep
    for k in range(threshold_rank):
        keep[order[k]] = True
    return keep


def fdr_filter_zscores(
    chunk_ids: list[str],
    zscores: list[float],
    *,
    alpha: float = 0.10,
) -> list[str]:
    pvals = [_normal_sf(float(z)) for z in zscores]
    keep_mask = benjamini_hochberg(pvals, alpha=alpha)
    return [cid for cid, keep in zip(chunk_ids, keep_mask, strict=True) if keep]
