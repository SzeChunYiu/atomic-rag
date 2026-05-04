"""Tests for the per-role rank-distribution diagnostic."""

from __future__ import annotations

from astro_cs_rag.benchmarks.evidence_crowding.generator import build_dataset
from astro_cs_rag.benchmarks.evidence_crowding.runner import EmbeddingCache
from astro_cs_rag.benchmarks.evidence_crowding.schema import CrowdingCell
from astro_cs_rag.diagnostics.rank_distribution import compute, to_dict
from astro_cs_rag.indexing.embedders import HashEmbedder, TrigramEmbedder


def _cell(seed: int = 7) -> CrowdingCell:
    return CrowdingCell(
        cell_id="t", n_distractors_per_gold=0, semantic_similarity="medium",
        entity_overlap="partial", answer_type_overlap=True,
        chunk_size=384, chunk_mixing="bridge_buried", hop_count=2,
        token_budget=1024, seed=seed,
    )


def test_rank_distribution_shape_and_serialise():
    ds = build_dataset(_cell(), n_queries=8)
    cache = EmbeddingCache(TrigramEmbedder())
    cache.build(ds)
    cd = compute(ds, cache)
    assert cd.n_atoms == len(ds.atoms)
    assert cd.n_queries == 8
    roles = {s.role for s in cd.by_role}
    assert "hop1" in roles and "hop2" in roles
    d = to_dict(cd)
    assert d["cell_id"] == ds.cell.cell_id
    assert all("median_global_rank" in s for s in d["by_role"])


def test_trigram_beats_hash_for_hop1_rank():
    """Hop1 shares rare film-name trigrams with the query — TrigramEmbedder
    should rank it dramatically better than the content-blind HashEmbedder.
    """
    ds = build_dataset(_cell(), n_queries=12)
    cache_hash = EmbeddingCache(HashEmbedder())
    cache_hash.build(ds)
    cd_hash = compute(ds, cache_hash)
    cache_tri = EmbeddingCache(TrigramEmbedder())
    cache_tri.build(ds)
    cd_tri = compute(ds, cache_tri)

    def hop1(cd):
        return next(s for s in cd.by_role if s.role == "hop1").median_global_rank

    assert hop1(cd_tri) < hop1(cd_hash) - 2, (hop1(cd_tri), hop1(cd_hash))


def test_within_role_rank_lower_than_global():
    """Within-role rank can never exceed global rank (subset)."""
    ds = build_dataset(_cell(), n_queries=10)
    cache = EmbeddingCache(TrigramEmbedder())
    cache.build(ds)
    cd = compute(ds, cache)
    for s in cd.by_role:
        assert s.median_within_role_rank <= s.median_global_rank + 1e-9
