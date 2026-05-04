"""Cherenkov-threshold candidate cut.

In a Cherenkov detector, only particles with v > c/n produce light. We import
the same operator: only candidates with relevance above a calibrated threshold
"emit" usable evidence; sub-threshold candidates contribute exactly zero, no
soft top-k tail.

The threshold is calibrated *per query* from the score distribution: for
example, the median + k * MAD ('robust z') of the candidate scores. This
adapts to per-query difficulty without a global hyper-parameter.
"""

from __future__ import annotations

import statistics


def cherenkov_threshold(
    scores: list[float],
    *,
    method: str = "median_mad",
    k: float = 2.0,
    quantile: float = 0.75,
) -> float:
    """Return the threshold below which candidates are rejected."""
    if not scores:
        return 0.0
    if method == "median_mad":
        med = float(statistics.median(scores))
        mad = float(statistics.median([abs(s - med) for s in scores])) or 1e-9
        return med + k * 1.4826 * mad
    if method == "quantile":
        sorted_s = sorted(scores)
        idx = int(len(sorted_s) * quantile)
        idx = min(max(idx, 0), len(sorted_s) - 1)
        return float(sorted_s[idx])
    if method == "fixed":
        return float(k)
    msg = f"unknown threshold method: {method}"
    raise ValueError(msg)


def cherenkov_filter(
    candidates: list[tuple[str, float]],
    *,
    method: str = "median_mad",
    k: float = 2.0,
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    thr = cherenkov_threshold([s for _, s in candidates], method=method, k=k)
    return [(cid, s) for cid, s in candidates if s >= thr]
