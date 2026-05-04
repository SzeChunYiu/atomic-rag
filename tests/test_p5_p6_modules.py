"""Smoke tests for P5/P6 inference modules."""

from __future__ import annotations

import numpy as np

from astro_cs_rag.detection.unfolding import unfold_scores
from astro_cs_rag.diagnostics.ood_gate import fit_gaussian_ood
from astro_cs_rag.diagnostics.tda import persistence_h0, tda_summary
from astro_cs_rag.retrieval.wasserstein import sinkhorn, wasserstein_score
from astro_cs_rag.selection.sbi import (
    CandidateSummary,
    featurize_subset,
    simulate_subsets,
    stand_in_posterior_score,
)
from astro_cs_rag.selection.smc import smc_select


def _orth(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d))
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m.astype(np.float32)


def test_sinkhorn_runs_and_returns_finite_cost() -> None:
    a_emb = _orth(4, seed=1)
    b_emb = _orth(5, seed=2)
    s = wasserstein_score(a_emb, b_emb)
    assert np.isfinite(s)


def test_sinkhorn_matches_uniform_weights() -> None:
    cost = np.zeros((3, 3), dtype=np.float32)
    a = np.ones(3, dtype=np.float64) / 3
    b = np.ones(3, dtype=np.float64) / 3
    P, ot = sinkhorn(cost, a, b, epsilon=0.1, n_iters=20)
    assert ot >= 0.0
    assert abs(P.sum() - 1.0) < 1e-6


def test_persistence_h0_returns_n_minus_1_bars() -> None:
    sims = np.array([[1, 0.9, 0.1], [0.9, 1, 0.2], [0.1, 0.2, 1]], dtype=np.float32)
    bars = persistence_h0(sims)
    assert len(bars) == 2  # n=3 → n-1=2 merges


def test_tda_summary_handles_small_input() -> None:
    embs = _orth(3, seed=4)
    score_field = {"a": 0.9, "b": 0.5, "c": 0.1}
    emb_by_id = {"a": embs[0], "b": embs[1], "c": embs[2]}
    out = tda_summary(score_field, emb_by_id)
    assert "mean_persistence" in out


def test_ood_gate_flags_distant_query() -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((100, 8)).astype(np.float32)
    model = fit_gaussian_ood(train, alpha=0.05)
    far = np.full((8,), 10.0, dtype=np.float32)
    assert model.is_ood(far)


def test_unfolding_smooths_noisy_score() -> None:
    embs = _orth(4, seed=5)
    raw = {f"c{i}": float(s) for i, s in enumerate([0.9, 0.1, 0.2, 0.85])}
    emb_by_id = {f"c{i}": embs[i] for i in range(4)}
    smoothed = unfold_scores(list(raw), raw, emb_by_id, sigma=0.5, laplacian_lambda=0.5)
    assert set(smoothed) == set(raw)
    # Smoothing pulls neighbor scores closer.
    assert max(smoothed.values()) <= max(raw.values())


def test_sbi_features_have_expected_dim() -> None:
    embs = _orth(3, seed=6)
    cands = [
        CandidateSummary(chunk_id=f"c{i}", snr=float(i), token_count=20, embedding=embs[i])
        for i in range(3)
    ]
    feat = featurize_subset(cands)
    assert feat.shape == (10,)


def test_simulate_subsets_respects_budget() -> None:
    cands = [CandidateSummary(chunk_id=f"c{i}", snr=1.0, token_count=80, embedding=None) for i in range(5)]
    subsets = simulate_subsets(cands, n_simulations=4, token_budget=200, seed=0)
    assert len(subsets) == 4
    for s in subsets:
        assert sum(c.token_count for c in s) <= 200


def test_smc_returns_ordered_selection() -> None:
    cands = [(f"c{i}", float(5 - i), 50) for i in range(5)]
    out = smc_select(cands, n_particles=8, token_budget=200, n_steps=3, seed=42)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)


def test_stand_in_posterior_score() -> None:
    embs = _orth(2, seed=8)
    cands_lo = [CandidateSummary(chunk_id="x", snr=0.1, token_count=10, embedding=embs[0])]
    cands_hi = [CandidateSummary(chunk_id="y", snr=5.0, token_count=10, embedding=embs[1])]
    assert stand_in_posterior_score(cands_hi) > stand_in_posterior_score(cands_lo)
