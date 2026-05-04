"""Local background estimates from score tails or global statistics."""

from __future__ import annotations

import statistics


def global_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mu = float(statistics.mean(values))
    sigma = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mu, sigma


def tail_mean_std(values: list[float], window: int) -> tuple[float, float]:
    if window <= 0:
        msg = "window must be positive"
        raise ValueError(msg)
    if not values:
        return 0.0, 0.0
    tail = values[-min(window, len(values)) :]
    mu = float(statistics.mean(tail))
    sigma = float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0
    return mu, sigma
