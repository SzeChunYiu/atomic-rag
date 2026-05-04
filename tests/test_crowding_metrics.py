"""Unit tests for crowding/blending/phase-transition diagnostics."""

from __future__ import annotations

import numpy as np

from astro_cs_rag.diagnostics.blending_metrics import (
    blending_gap,
    coarsening_loss,
    cosine,
    per_chunk_blending_table,
)
from astro_cs_rag.diagnostics.crowding_metrics import (
    all_gold_recall_at_k,
    chunk_crowding,
    dense_crowding,
    entity_crowding,
    support_chain_complete,
    type_crowding,
)
from astro_cs_rag.diagnostics.phase_transition import (
    auc_success,
    estimate_threshold,
    fit,
    slope_at,
)


def test_cosine_basic():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine(a, a) == 1.0
    assert cosine(a, b) == 0.0
    assert cosine(np.zeros(2), a) == 0.0


def test_blending_gap_positive_when_atom_beats_chunk():
    q = np.array([1.0, 0.0])
    chunk = np.array([0.5, 1.0])  # noisy chunk pooling
    atoms = np.array([[1.0, 0.1], [0.0, 1.0]])
    gap = blending_gap(q, chunk, atoms)
    assert gap > 0


def test_blending_gap_zero_when_no_atoms():
    q = np.array([1.0, 0.0])
    chunk = np.array([1.0, 0.0])
    assert blending_gap(q, chunk, np.empty((0, 2))) == 0.0


def test_per_chunk_blending_table_shape_and_seen_mask():
    q = np.array([1.0, 0.0])
    chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    atoms = np.array([[1.0, 0.0], [0.0, 1.0]])
    a2c = np.array([0, 1])
    gaps = per_chunk_blending_table(q, chunks, atoms, a2c)
    assert gaps.shape == (3,)
    assert gaps[2] == 0.0  # no atoms in chunk 2


def test_coarsening_loss_sign():
    assert abs(coarsening_loss(0.9, 0.7) - 0.2) < 1e-9
    assert coarsening_loss(0.5, 0.6) < 0


def test_dense_crowding_excludes_self_and_other_gold():
    e = np.array([
        [1.0, 0.0],
        [0.99, 0.05],
        [0.0, 1.0],
        [0.95, 0.10],
    ])
    n = dense_crowding(0, e, gold_atom_idxs={0, 3}, radius=0.05)
    assert n == 1  # only atom 1 is a non-gold neighbor


def test_type_and_entity_and_chunk_crowding():
    assert type_crowding("WHERE", ["WHERE", "WHEN", "WHERE", "WHO"], {0}) == 1
    n_ent = entity_crowding(
        query_entities={"Norway"},
        gold_entities={"Mara"},
        candidate_entities=[{"Mara", "Oslo"}, {"Bob"}, {"Norway"}, set()],
        gold_idxs_in_candidates={0},
    )
    assert n_ent == 1  # candidate 2 shares "Norway"
    assert chunk_crowding("c1", ["c1", "c1", "c2"], {0}) == 1


def test_support_chain_and_recall():
    assert support_chain_complete(["a", "b", "c"], ["a", "b"]) is True
    assert support_chain_complete(["a", "c"], ["a", "b"]) is False
    assert support_chain_complete([], []) is False
    assert all_gold_recall_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0
    assert all_gold_recall_at_k(["a", "b", "c"], ["a", "b"], k=1) == 0.5


def test_phase_threshold_and_auc():
    xs = [0.0, 5.0, 10.0, 20.0, 50.0]
    ys = [0.95, 0.80, 0.55, 0.30, 0.10]
    thr = estimate_threshold(xs, ys, cutoff=0.5)
    assert thr is not None and 10.0 < thr < 20.0
    assert 0.0 < auc_success(xs, ys) < 1.0
    f = fit(xs, ys)
    assert f.slope_at_threshold < 0  # success drops with crowding
    assert f.n_points == 5


def test_phase_threshold_returns_none_when_never_crosses():
    xs = [0.0, 5.0, 10.0]
    ys = [0.9, 0.85, 0.80]
    assert estimate_threshold(xs, ys) is None
    assert slope_at(xs, ys, None) == 0.0
