"""Score-field morphology — borrowed from calorimeter shower-shape analysis.

For each query, the retrieval score field s(i) over the corpus has a *shape*.
We summarize that shape with a small set of statistics and project queries
into an archetype space (compact, plateau, bimodal, noisy). Later phases
route different physics methods to different archetypes.

This module is method-agnostic: it consumes a ranked score list, produces
a ScoreShape record. It has no dependency on retrievers or LLMs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreShape:
    """Calorimetric summary of a 1-D score field over candidates.

    Fields are dimensionless / robust where possible so they compare across
    retrievers and corpora.
    """

    n_candidates: int
    peak_score: float
    peak_minus_median: float            # depth of the leading peak above bulk
    fwhm_index_fraction: float          # full-width-at-half-max as a fraction of N
    kurtosis: float                     # heavy-tailed-ness of the score distribution
    skewness: float                     # asymmetry
    bimodality_coefficient: float       # SAS bimodality coefficient (>5/9 ~ bimodal)
    second_peak_gap: float              # (peak1 - peak2) / peak1, robust to outliers
    tail_decay_slope: float             # slope of log-score vs log-rank (Zipf exponent)
    archetype: str                      # discrete label (compact|plateau|bimodal|noisy|empty)


def _safe_std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=0))


def _safe_skew_kurt(x: np.ndarray) -> tuple[float, float]:
    if x.size < 4:
        return 0.0, 0.0
    mu = float(np.mean(x))
    sigma = _safe_std(x)
    if sigma <= 1e-12:
        return 0.0, 0.0
    z = (x - mu) / sigma
    skew = float(np.mean(z**3))
    # excess kurtosis (Fisher).
    kurt = float(np.mean(z**4) - 3.0)
    return skew, kurt


def _bimodality_coefficient(skew: float, kurt: float, n: int) -> float:
    """SAS bimodality coefficient. >5/9 ~ bimodal."""
    if n < 4:
        return 0.0
    num = skew * skew + 1.0
    denom_corr = 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else 0.0
    denom = max(kurt + denom_corr, 1e-9)
    return float(num / denom)


def _fwhm_fraction(scores_desc: np.ndarray) -> float:
    if scores_desc.size <= 1:
        return 0.0
    peak = float(scores_desc[0])
    floor = float(np.median(scores_desc))
    half = floor + 0.5 * (peak - floor)
    above = int(np.sum(scores_desc >= half))
    return float(above) / float(scores_desc.size)


def _second_peak_gap(scores_desc: np.ndarray) -> float:
    if scores_desc.size < 2:
        return 0.0
    peak = float(scores_desc[0])
    if peak <= 1e-12:
        return 0.0
    runner = float(scores_desc[1])
    return float((peak - runner) / max(peak, 1e-12))


def _tail_slope(scores_desc: np.ndarray) -> float:
    """Fit log(score) ~ a + b*log(rank); return b. Negative b = fast decay."""
    n = scores_desc.size
    if n < 4:
        return 0.0
    s = scores_desc - float(np.min(scores_desc)) + 1e-9
    s = np.maximum(s, 1e-12)
    ranks = np.arange(1, n + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(s)
    a, b = np.polyfit(x, y, 1)
    return float(a)


def _classify(
    *,
    fwhm_frac: float,
    bimod: float,
    second_gap: float,
    peak_minus_med: float,
    n: int,
) -> str:
    if n == 0:
        return "empty"
    if peak_minus_med < 1e-9:
        return "noisy"
    if bimod > 5.0 / 9.0 and second_gap < 0.4:
        return "bimodal"
    if fwhm_frac < 0.05 and second_gap > 0.5:
        return "compact"
    if fwhm_frac > 0.30:
        return "plateau"
    return "diffuse"


def score_shape(scores: list[float]) -> ScoreShape:
    arr = np.asarray(sorted(scores, reverse=True), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return ScoreShape(
            n_candidates=0,
            peak_score=0.0,
            peak_minus_median=0.0,
            fwhm_index_fraction=0.0,
            kurtosis=0.0,
            skewness=0.0,
            bimodality_coefficient=0.0,
            second_peak_gap=0.0,
            tail_decay_slope=0.0,
            archetype="empty",
        )
    peak = float(arr[0])
    median = float(np.median(arr))
    skew, kurt = _safe_skew_kurt(arr)
    bimod = _bimodality_coefficient(skew, kurt, n)
    fwhm = _fwhm_fraction(arr)
    gap = _second_peak_gap(arr)
    slope = _tail_slope(arr)
    arch = _classify(
        fwhm_frac=fwhm,
        bimod=bimod,
        second_gap=gap,
        peak_minus_med=peak - median,
        n=n,
    )
    return ScoreShape(
        n_candidates=n,
        peak_score=peak,
        peak_minus_median=peak - median,
        fwhm_index_fraction=fwhm,
        kurtosis=float(kurt) if math.isfinite(kurt) else 0.0,
        skewness=float(skew) if math.isfinite(skew) else 0.0,
        bimodality_coefficient=float(bimod) if math.isfinite(bimod) else 0.0,
        second_peak_gap=gap,
        tail_decay_slope=slope,
        archetype=arch,
    )
