"""Phase-transition analysis for crowding sweeps.

Given P(success) measured at a sequence of crowding levels, estimate:
- C\\* — the smallest crowding level where P(success) crosses below 0.5.
- AUC — area under the success curve.
- slope — local logistic slope at the threshold.

These are reported per system and per crowding axis. A sharp slope is
evidence of a phase transition; a shallow slope is smooth degradation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseFit:
    threshold: float | None
    auc: float
    slope_at_threshold: float
    n_points: int


def estimate_threshold(
    crowding_levels: list[float],
    success_rates: list[float],
    cutoff: float = 0.5,
) -> float | None:
    """Linear interpolation of the first downward crossing of `cutoff`.

    Returns None if every measured rate stays above (or stays below) the
    cutoff over the swept range.
    """
    if len(crowding_levels) != len(success_rates) or not crowding_levels:
        return None
    pts = sorted(zip(crowding_levels, success_rates), key=lambda p: p[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= cutoff and y1 < cutoff:
            if y0 == y1:
                return x0
            t = (cutoff - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    if pts[0][1] < cutoff:
        return float(pts[0][0])
    return None


def auc_success(
    crowding_levels: list[float],
    success_rates: list[float],
) -> float:
    """Trapezoidal AUC normalised by the swept range. Scale-free in [0, 1]."""
    if len(crowding_levels) < 2:
        return 0.0
    x = np.array(crowding_levels, dtype=np.float64)
    y = np.array(success_rates, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    if span <= 0:
        return float(y.mean())
    return float(np.trapezoid(y, x) / span)


def slope_at(
    crowding_levels: list[float],
    success_rates: list[float],
    threshold: float | None,
) -> float:
    """Approximate dP/dx at `threshold` by central difference on neighbors.

    Returns 0.0 if the threshold is undefined or if there are fewer than
    two distinct crowding levels.
    """
    if threshold is None or len(crowding_levels) < 2:
        return 0.0
    pts = sorted(zip(crowding_levels, success_rates), key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for i in range(len(xs) - 1):
        if xs[i] <= threshold <= xs[i + 1] and xs[i + 1] != xs[i]:
            return float((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
    return 0.0


def fit(
    crowding_levels: list[float],
    success_rates: list[float],
    cutoff: float = 0.5,
) -> PhaseFit:
    thr = estimate_threshold(crowding_levels, success_rates, cutoff)
    return PhaseFit(
        threshold=thr,
        auc=auc_success(crowding_levels, success_rates),
        slope_at_threshold=slope_at(crowding_levels, success_rates, thr),
        n_points=len(crowding_levels),
    )
