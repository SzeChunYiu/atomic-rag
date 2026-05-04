"""Smoke tests for the evidence-crowding generator + runner.

These keep the synthetic benchmark honest:
  - schema invariants (gold atoms tagged, doc_ids populated, chunks
    accounted for)
  - crowding monotonicity: P(support_chain_complete) decreases as
    n_distractors_per_gold grows, for both registered baselines.
"""

from __future__ import annotations

from astro_cs_rag.benchmarks.evidence_crowding.generator import build_dataset
from astro_cs_rag.benchmarks.evidence_crowding.runner import run_cell
from astro_cs_rag.benchmarks.evidence_crowding.schema import CrowdingCell
from astro_cs_rag.benchmarks.evidence_crowding.sweeps import smoke_grid


def _cell(nd: int, seed: int = 7) -> CrowdingCell:
    return CrowdingCell(
        cell_id=f"t_nd{nd}",
        n_distractors_per_gold=nd,
        semantic_similarity="medium",
        entity_overlap="partial",
        answer_type_overlap=True,
        chunk_size=384,
        chunk_mixing="bridge_buried",
        hop_count=2,
        token_budget=256,
        seed=seed,
    )


def test_semantic_similarity_drives_monotone_cos_to_query():
    """`semantic_similarity` must produce a monotone shift in
    cos(distractor, query) for the axis to be meaningful for retrieval.
    Round 4 wired the paraphrase pool but left high == medium in
    cos-to-query; round 5 controls the distractor class-mix so that
    high distractors share the *query's distinguishing entity* (film).
    """
    import numpy as np

    from astro_cs_rag.benchmarks.evidence_crowding.runner import EmbeddingCache
    from astro_cs_rag.indexing.embedders import TrigramEmbedder

    def mean_cos_to_query(sim: str) -> float:
        cell = CrowdingCell(
            cell_id=f"m_{sim}", n_distractors_per_gold=8,
            semantic_similarity=sim, entity_overlap="partial",
            answer_type_overlap=True, chunk_size=384,
            chunk_mixing="bridge_buried", hop_count=2,
            token_budget=1024, seed=2026,
        )
        ds = build_dataset(cell, n_queries=12)
        cache = EmbeddingCache(TrigramEmbedder())
        cache.build(ds)
        cs: list[float] = []
        for q in ds.queries:
            qv = cache.query_emb[q.query_id]
            qn = qv / (np.linalg.norm(qv) + 1e-12)
            for a in ds.atoms:
                if a.role != "distractor" or not a.atom_id.startswith(q.query_id):
                    continue
                v = cache.atom_emb[cache.atom_idx[a.atom_id]]
                vn = v / (np.linalg.norm(v) + 1e-12)
                cs.append(float(qn @ vn))
        return float(np.mean(cs))

    low = mean_cos_to_query("low")
    med = mean_cos_to_query("medium")
    high = mean_cos_to_query("high")
    assert low < med < high, (low, med, high)
    # And the gap should be substantial — not a noise-level shift.
    assert high - low > 0.10, (low, med, high)


def test_dataset_schema_invariants():
    ds = build_dataset(_cell(nd=5), n_queries=4)
    assert len(ds.queries) == 4
    gold_atoms = [a for a in ds.atoms if a.is_gold]
    # 2 gold atoms (hop1, hop2) per query
    assert len(gold_atoms) == 8
    # all atoms have a chunk and doc id assigned after packing
    assert all(a.chunk_id and a.doc_id for a in ds.atoms)
    chunk_ids = {c.chunk_id for c in ds.chunks}
    assert {a.chunk_id for a in ds.atoms} <= chunk_ids
    # gold_doc_ids populated per query, and the gold atoms live in those docs
    by_atom = {a.atom_id: a for a in ds.atoms}
    for q in ds.queries:
        assert q.gold_doc_ids
        assert {by_atom[aid].doc_id for aid in q.gold_atom_ids} == set(q.gold_doc_ids)


def test_smoke_grid_runs_both_systems():
    cells = list(smoke_grid())
    assert len(cells) == 3
    rows = []
    for cell in cells:
        ds = build_dataset(cell, n_queries=6)
        rows.extend(run_cell(ds, systems=["atom_dense", "chunk_dense"]))
    systems = {r.system_name for r in rows}
    assert systems == {"atom_dense", "chunk_dense"}
    # 3 cells * 6 queries * 2 systems
    assert len(rows) == 36


def test_iter2_beats_dense_at_low_distractors_constrained_budget():
    """When budget is tight and distractors are few, expanding the query
    with the top-1 atom should let hop2 reach the budget. At higher nd
    this advantage collapses — also part of the published finding, not
    asserted here.
    """

    def rate(nd: int, system: str, tb: int) -> float:
        ds = build_dataset(_cell(nd=nd), n_queries=24)
        rows = run_cell(ds, systems=[system])
        return sum(r.support_chain_complete for r in rows) / len(rows)

    dense_at_nd0_tb256 = rate(0, "atom_dense", 256)
    iter2_at_nd0_tb256 = rate(0, "atom_iter2", 256)
    assert iter2_at_nd0_tb256 > dense_at_nd0_tb256 + 0.10, (
        dense_at_nd0_tb256, iter2_at_nd0_tb256
    )


def test_crowding_degrades_support_chain_completion():
    """As n_distractors grows, support-chain completion must not increase."""

    def rate(nd: int, system: str) -> float:
        ds = build_dataset(_cell(nd=nd), n_queries=12)
        rows = run_cell(ds, systems=[system])
        return sum(r.support_chain_complete for r in rows) / len(rows)

    for sysname in ("atom_dense", "chunk_dense"):
        r0 = rate(0, sysname)
        r_high = rate(50, sysname)
        # zero distractors should be (near-)perfect; heavy crowding strictly worse
        assert r0 >= 0.9, (sysname, r0)
        assert r_high < r0, (sysname, r0, r_high)
