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


def test_semantic_similarity_axis_actually_moves_distractor_text():
    """`cell.semantic_similarity` was previously declared but unused —
    every cell shared the same paraphrase distribution. After round 4
    the distractor pool is distinct per level: 'high' uses the gold
    paraphrase verbatim, 'low' uses distant paraphrases.
    """

    def distractor_texts(sim: str) -> set[str]:
        cell = CrowdingCell(
            cell_id=f"sim_{sim}", n_distractors_per_gold=4,
            semantic_similarity=sim, entity_overlap="partial",
            answer_type_overlap=True, chunk_size=384,
            chunk_mixing="bridge_buried", hop_count=2,
            token_budget=1024, seed=99,
        )
        ds = build_dataset(cell, n_queries=8)
        return {a.text for a in ds.atoms if a.role == "distractor"}

    low, med, high = (distractor_texts(s) for s in ("low", "medium", "high"))
    assert low != med, "low and medium produce identical distractors — axis is dead"
    assert high != med, "high and medium produce identical distractors — axis is dead"
    assert any("biography" in t or "early years" in t or "childhood" in t for t in low)


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
