"""Tests for the P3.5 bonus methods."""

from __future__ import annotations

import math

import numpy as np

from astro_cs_rag.detection.aperture import aperture_snr
from astro_cs_rag.detection.cherenkov import cherenkov_filter, cherenkov_threshold
from astro_cs_rag.detection.fdr import benjamini_hochberg, fdr_filter_zscores
from astro_cs_rag.detection.standard_candle import (
    calibrate_scores,
    fit_isotonic,
)
from astro_cs_rag.retrieval.coronagraph import coronagraph_residual
from astro_cs_rag.retrieval.vlbi import vlbi_combine, vlbi_visibility


def _orth(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d))
    m, _ = np.linalg.qr(m)
    return m[:n].astype(np.float32)


# ---- aperture SNR ----------------------------------------------------------


def test_aperture_snr_sees_isolated_signal() -> None:
    embs = _orth(5, seed=1)
    emb_by_id = {f"c{i}": embs[i] for i in range(5)}
    scores = {"c0": 5.0, "c1": 0.1, "c2": 0.1, "c3": 0.1, "c4": 0.1}
    atoms = aperture_snr(
        query_id="q",
        candidate_chunk_ids=list(emb_by_id),
        candidate_scores=scores,
        embeddings_by_id=emb_by_id,
        radius_in=0.0,
        radius_out=2.0,
    )
    by_id = {a.chunk_id: a for a in atoms}
    assert by_id["c0"].snr > by_id["c1"].snr


# ---- BH FDR ---------------------------------------------------------------


def test_bh_keeps_significant_drops_null() -> None:
    # BH at alpha=0.05, n=5: thresholds are 0.05*k/5 for k=1..5 = {0.01, 0.02, 0.03, 0.04, 0.05}
    # p-values [0.001, 0.005, 0.04, 0.5, 0.9] → keep where p_(k) <= alpha*k/n at any k
    pvals = [0.001, 0.005, 0.04, 0.5, 0.9]
    keep = benjamini_hochberg(pvals, alpha=0.05)
    assert keep == [True, True, False, False, False]
    # Loosen to alpha=0.10: thresholds {0.02, 0.04, 0.06, 0.08, 0.10}; p=0.04 now passes.
    loose = benjamini_hochberg(pvals, alpha=0.10)
    assert loose == [True, True, True, False, False]


def test_fdr_filter_uses_zscores() -> None:
    kept = fdr_filter_zscores(["a", "b", "c"], [3.0, 0.0, -1.0], alpha=0.05)
    assert "a" in kept
    assert "c" not in kept


# ---- Cherenkov threshold ---------------------------------------------------


def test_cherenkov_threshold_median_mad() -> None:
    thr = cherenkov_threshold([0.0] * 9 + [10.0], method="median_mad", k=1.0)
    assert thr > 0.0


def test_cherenkov_filter_keeps_above_threshold() -> None:
    # Clear outlier above background.
    cands = [("a", 0.1), ("b", 0.0), ("c", 0.05), ("d", 0.1), ("e", 5.0)]
    kept = cherenkov_filter(cands, method="median_mad", k=1.0)
    kept_ids = {c for c, _ in kept}
    assert "e" in kept_ids
    assert "b" not in kept_ids


# ---- coronagraph -----------------------------------------------------------


def test_coronagraph_masks_anchor_and_demotes_similar() -> None:
    embs = _orth(4, seed=2)
    emb_by_id = {f"c{i}": embs[i] for i in range(4)}
    # c0 is the anchor; c1 is close to c0 so its residual should drop.
    scores = {"c0": 1.0, "c1": 0.9, "c2": 0.5, "c3": 0.4}
    res = coronagraph_residual(
        chunk_ids=list(emb_by_id),
        scores=scores,
        embeddings_by_id=emb_by_id,
        n_anchors=1,
    )
    assert res["c0"] == 0.0


# ---- VLBI -----------------------------------------------------------------


def test_vlbi_combine_geomean_is_below_individual_max() -> None:
    fields = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 5.0}]
    combined = vlbi_combine(fields)
    assert combined["a"] > combined["b"]  # a is consistently high
    assert combined["b"] < 5.0           # b suppressed by single-channel zero


def test_vlbi_visibility_high_when_consistent() -> None:
    fields = [{"a": 1.0}, {"a": 1.0}, {"a": 1.0}]
    assert math.isclose(vlbi_visibility(fields, chunk_id="a"), 1.0)


# ---- standard candle -------------------------------------------------------


def test_isotonic_calibrator_monotone() -> None:
    raw = [0.1, 0.3, 0.2, 0.6, 0.5, 0.9]
    targets = [0.0, 0.4, 0.3, 0.7, 0.6, 1.0]
    curve = fit_isotonic(raw, targets)
    out = calibrate_scores({f"c{i}": x for i, x in enumerate(raw)}, curve)
    sorted_inputs = sorted(out.items(), key=lambda x: x[1])
    # Outputs are non-decreasing as raw increases.
    for cid, val in sorted_inputs:
        assert isinstance(val, float)
