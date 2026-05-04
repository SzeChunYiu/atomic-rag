"""Standard-candle calibration of the relevance scale.

Like Type Ia supernovae and Cepheids, we use *known-relevance* anchors to
convert relative retrieval scores into a calibrated absolute scale.
Practical recipe:

  1. Maintain a small *anchor set* of (query, gold_text) pairs whose true
     relevance == 1.0 by construction (Asimov-style).
  2. At retrieval time, run the anchor queries through the same retriever and
     measure their score-at-gold; this is the calibrator data.
  3. Fit a monotone map raw_score -> absolute_relevance via isotonic
     regression. Apply to all production scores.

The result is a relevance scale with operational meaning ("0.5 = roughly half
the certainty of a known-good gold passage"). Abstention thresholds become
interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationCurve:
    raw_breakpoints: list[float]
    abs_breakpoints: list[float]

    def apply(self, raw: float) -> float:
        if not self.raw_breakpoints:
            return float(raw)
        if raw <= self.raw_breakpoints[0]:
            return float(self.abs_breakpoints[0])
        if raw >= self.raw_breakpoints[-1]:
            return float(self.abs_breakpoints[-1])
        for i in range(len(self.raw_breakpoints) - 1):
            x0, x1 = self.raw_breakpoints[i], self.raw_breakpoints[i + 1]
            if x0 <= raw <= x1:
                t = (raw - x0) / max(x1 - x0, 1e-12)
                y0, y1 = self.abs_breakpoints[i], self.abs_breakpoints[i + 1]
                return float(y0 + t * (y1 - y0))
        return float(raw)


def fit_isotonic(raw_scores: list[float], abs_targets: list[float]) -> CalibrationCurve:
    """PAV-style isotonic regression — single-pass merge."""
    if len(raw_scores) != len(abs_targets):
        msg = "raw and target length mismatch"
        raise ValueError(msg)
    if not raw_scores:
        return CalibrationCurve(raw_breakpoints=[], abs_breakpoints=[])
    pairs = sorted(zip(raw_scores, abs_targets, strict=True))
    xs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]
    weights = [1.0] * len(ys)
    i = 0
    while i < len(ys) - 1:
        if ys[i] > ys[i + 1]:
            new_w = weights[i] + weights[i + 1]
            new_y = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / new_w
            ys[i] = new_y
            weights[i] = new_w
            del ys[i + 1]
            del weights[i + 1]
            del xs[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return CalibrationCurve(raw_breakpoints=xs, abs_breakpoints=ys)


def calibrate_scores(
    raw_scores_by_chunk: dict[str, float],
    curve: CalibrationCurve,
) -> dict[str, float]:
    return {cid: curve.apply(s) for cid, s in raw_scores_by_chunk.items()}
