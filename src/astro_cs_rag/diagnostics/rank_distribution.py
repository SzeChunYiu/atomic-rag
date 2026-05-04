"""Per-role atom-rank diagnostic for crowding cells.

Given a built EmbeddingCache + dataset, compute for every query:
  - the rank of each gold atom in the global cos-similarity ordering
  - the rank of each gold atom *among atoms that share its role*
  - the rank of the highest-scoring distractor

Then aggregate per role into median / mean / quantiles. This makes the
"why is hop2 ranked worse than random" question a one-line metric in
every sweep summary, not a one-off audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

import numpy as np


@dataclass
class RankSummary:
    role: str
    n: int
    median_global_rank: float
    mean_global_rank: float
    median_within_role_rank: float
    pct_in_top1: float
    pct_in_top5: float


@dataclass
class CellRankDistribution:
    cell_id: str
    n_atoms: int
    n_queries: int
    by_role: list[RankSummary] = field(default_factory=list)


def _rank(scores: np.ndarray, idx: int) -> int:
    """0-based rank of `idx` in descending order of `scores`."""
    return int(np.sum(scores > scores[idx]))


def compute(dataset, cache) -> CellRankDistribution:
    """Compute per-role rank distribution for one cell.

    `dataset`/`cache` are duck-typed to the evidence_crowding stack.
    """
    atoms = dataset.atoms
    role_idxs: dict[str, list[int]] = {}
    for i, a in enumerate(atoms):
        role_idxs.setdefault(a.role or "unknown", []).append(i)

    # gather (role, global_rank, within_role_rank) for every gold atom
    by_role_global: dict[str, list[int]] = {}
    by_role_within: dict[str, list[int]] = {}
    for q in dataset.queries:
        qv = cache.query_emb[q.query_id]
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        anorms = np.linalg.norm(cache.atom_emb, axis=1) + 1e-12
        scores = (cache.atom_emb @ qn) / anorms
        for aid in q.gold_atom_ids:
            ai = cache.atom_idx[aid]
            role = atoms[ai].role or "unknown"
            by_role_global.setdefault(role, []).append(_rank(scores, ai))
            peers = role_idxs.get(role, [])
            if peers:
                peer_scores = scores[peers]
                local_rank = int(np.sum(peer_scores > scores[ai]))
                by_role_within.setdefault(role, []).append(local_rank)

    summaries: list[RankSummary] = []
    n = len(atoms)
    for role in sorted(by_role_global):
        gr = by_role_global[role]
        wr = by_role_within.get(role, [])
        in_top1 = sum(1 for r in gr if r == 0) / max(1, len(gr))
        in_top5 = sum(1 for r in gr if r < 5) / max(1, len(gr))
        summaries.append(
            RankSummary(
                role=role,
                n=len(gr),
                median_global_rank=float(median(gr)) if gr else 0.0,
                mean_global_rank=float(sum(gr) / len(gr)) if gr else 0.0,
                median_within_role_rank=float(median(wr)) if wr else 0.0,
                pct_in_top1=in_top1,
                pct_in_top5=in_top5,
            )
        )
    return CellRankDistribution(
        cell_id=dataset.cell.cell_id,
        n_atoms=n,
        n_queries=len(dataset.queries),
        by_role=summaries,
    )


def to_dict(cd: CellRankDistribution) -> dict:
    return {
        "cell_id": cd.cell_id,
        "n_atoms": cd.n_atoms,
        "n_queries": cd.n_queries,
        "by_role": [s.__dict__ for s in cd.by_role],
    }
