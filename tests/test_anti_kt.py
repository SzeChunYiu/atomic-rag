"""Anti-kT clustering correctness + IRC-safety numerical tests."""

from __future__ import annotations

import numpy as np

from astro_cs_rag.selection.anti_kt import cluster_anti_kt, select_evidence_via_jets


def _orthonormal(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, d))
    m, _ = np.linalg.qr(m)
    return m[:n].astype(np.float32)


def test_clustering_returns_finalized_jets() -> None:
    emb = _orthonormal(4)
    res = cluster_anti_kt(["a", "b", "c", "d"], [3.0, 2.5, 0.5, 0.3], emb)
    assert len(res.final_jets) >= 1
    assert all(j.relevance > 0 for j in res.final_jets)


def test_collinear_safety_chunk_split_invariance() -> None:
    """Splitting a chunk into two co-located halves does not change the leading jet."""
    emb = _orthonormal(3, seed=1)
    base_ids = ["x", "y", "z"]
    base_rel = [3.0, 1.0, 0.4]
    leading_orig = set(select_evidence_via_jets(base_ids, base_rel, emb, R=0.5))

    # Split x into x1, x2 at the same direction with relevance 1.5 each.
    split_ids = ["x1", "x2", "y", "z"]
    split_rel = [1.5, 1.5, 1.0, 0.4]
    split_emb = np.stack([emb[0], emb[0], emb[1], emb[2]], axis=0)
    leading_split = set(select_evidence_via_jets(split_ids, split_rel, split_emb, R=0.5))
    # Leading-jet membership: original {x} → split {x1, x2}; both should appear.
    assert leading_split == {"x1", "x2"} or leading_orig <= leading_split.union({"x1", "x2"})


def test_infrared_safety_hard_constituents_preserved() -> None:
    """Adding arbitrarily soft distractors must not drop any hard constituent
    from the leading jet. (Soft atoms may be absorbed in; that's IR-safety.)"""
    emb = _orthonormal(3, seed=2)
    leading_a = set(select_evidence_via_jets(["x", "y", "z"], [3.0, 1.0, 0.5], emb, R=0.7))

    n_extra = 5
    rng = np.random.default_rng(3)
    extra_emb = rng.standard_normal((n_extra, emb.shape[1])).astype(np.float32)
    extra_emb /= np.linalg.norm(extra_emb, axis=1, keepdims=True) + 1e-9

    full_emb = np.vstack([emb, extra_emb])
    full_ids = ["x", "y", "z", *[f"d{i}" for i in range(n_extra)]]
    full_rel = [3.0, 1.0, 0.5, *[1e-4] * n_extra]
    leading_b = set(select_evidence_via_jets(full_ids, full_rel, full_emb, R=0.7))
    # Hard constituents must survive; soft atoms may be absorbed into the jet.
    assert leading_a.issubset(leading_b)


def test_smaller_R_produces_more_jets() -> None:
    emb = _orthonormal(6, seed=4)
    rel = [1.0, 0.95, 0.9, 0.4, 0.3, 0.2]
    big_R = cluster_anti_kt(["a", "b", "c", "d", "e", "f"], rel, emb, R=2.0)
    small_R = cluster_anti_kt(["a", "b", "c", "d", "e", "f"], rel, emb, R=0.2)
    assert len(small_R.final_jets) >= len(big_R.final_jets)


def test_empty_input_returns_empty_result() -> None:
    res = cluster_anti_kt([], [], np.zeros((0, 4), dtype=np.float32))
    assert res.final_jets == []
    assert res.leading_atoms == []
