"""Pre-flight check: does the embedder distinguish gold from random?

Round 1 of the D04 study burned compute on HashEmbedder, whose
sha256-of-full-text vectors carry no token-level signal — gold atoms
ranked uniformly at random. The garbage was only obvious after the
sweep finished. This check runs the same audit before a sweep starts
and refuses to proceed if cos(query, gold_hop1) doesn't outperform
random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astro_cs_rag.benchmarks.evidence_crowding.generator import build_dataset
from astro_cs_rag.benchmarks.evidence_crowding.runner import EmbeddingCache
from astro_cs_rag.benchmarks.evidence_crowding.schema import CrowdingCell


@dataclass(frozen=True)
class EmbedderCheck:
    n_atoms: int
    n_queries: int
    median_hop1_rank: float
    random_expected_rank: float
    z_score: float
    passed: bool
    reason: str


def _probe_cell(seed: int = 0) -> CrowdingCell:
    return CrowdingCell(
        cell_id=f"probe_{seed}", n_distractors_per_gold=2,
        semantic_similarity="medium", entity_overlap="partial",
        answer_type_overlap=True, chunk_size=384,
        chunk_mixing="bridge_buried", hop_count=2,
        token_budget=1024, seed=seed,
    )


def check(embedder, n_queries: int = 32, seed: int = 0) -> EmbedderCheck:
    """Run the audit; pass if hop1 ranks decisively above random.

    Two conditions, both required:
      - median hop1 rank in the top 25% of the corpus
      - z-score against a uniform-rank null exceeds 4
    Hash-style embedders typically fail both; real embedders cross by
    a wide margin (median rank near 0).
    """
    ds = build_dataset(_probe_cell(seed), n_queries=n_queries)
    cache = EmbeddingCache(embedder)
    cache.build(ds)
    n = len(cache.atom_ids)
    ranks: list[int] = []
    for q in ds.queries:
        qv = cache.query_emb[q.query_id]
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        anorms = np.linalg.norm(cache.atom_emb, axis=1) + 1e-12
        scores = (cache.atom_emb @ qn) / anorms
        for aid in q.gold_atom_ids:
            if "hop1" in aid:
                ai = cache.atom_idx[aid]
                ranks.append(int(np.sum(scores > scores[ai])))
    med = float(np.median(ranks))
    expected = (n - 1) / 2.0
    # Variance of rank under uniform-random null = (N^2 - 1)/12.
    sd_one = ((n * n - 1) / 12.0) ** 0.5
    sd_med = sd_one / (len(ranks) ** 0.5)
    z = (expected - med) / max(1e-9, sd_med)
    passed = med < 0.25 * n and z > 4.0
    reason = (
        f"hop1 median rank {med:.1f} vs random expected {expected:.1f} "
        f"(z={z:.2f} over {len(ranks)} probes)"
    )
    if not passed:
        reason += " — embedder appears content-blind"
    return EmbedderCheck(
        n_atoms=n, n_queries=n_queries,
        median_hop1_rank=med, random_expected_rank=expected,
        z_score=z, passed=passed, reason=reason,
    )
