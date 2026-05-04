"""Cross-section formalism for retrieval evaluation.

Borrowed from particle physics:  N_obs = σ · L · ε
- N_obs : observed gold-acquisition events (a chunk-level gold hit at top-k).
- L     : query luminosity = number of queries × candidate budget.
- ε     : downstream selection / generation efficiency (factored separately).
- σ     : retriever cross section = intrinsic per-query gold-acquisition rate
          *normalized* by luminosity. This isolates retriever quality from
          corpus / query mix.

Operationally we report two scalars per method:
- σ_recall@k    = Σ_q recall@k(q) / Σ_q L_q,  where  L_q = min(|gold_q|, k)
                  → matches recall@k mean for queries that have ≤ k golds, but
                  divides out trivial luminosity differences across query mixes.
- σ_efficiency  = (mean answer EM) / (mean recall@k)   → generation efficiency.

These two together let us decompose the full pipeline as:
  EM ≈ σ_recall@k × σ_efficiency.

Honest framing: both σ values are dimensionless and each depends on k. The
formalism's value is the *additivity* / *factorization* — not a single magic
number — and the ability to compare retrievers across heterogeneous corpora
without confounding from query luminosity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CrossSectionRow:
    method: str
    dataset: str
    k: int
    n_queries: int
    luminosity: float
    sigma_recall: float
    sigma_efficiency: float | None


def compute_sigma_recall(
    *,
    per_query_recall: Mapping[str, float],
    per_query_n_gold: Mapping[str, int],
    k: int,
) -> tuple[float, float, int]:
    """Return (σ_recall, total_luminosity, n_queries_used)."""
    n = 0
    total_recall = 0.0
    total_lumi = 0.0
    for qid, rec in per_query_recall.items():
        n_gold = int(per_query_n_gold.get(qid, 0))
        if n_gold == 0:
            continue
        lumi = float(min(n_gold, k))
        total_recall += float(rec) * lumi
        total_lumi += lumi
        n += 1
    if total_lumi <= 0.0 or n == 0:
        return 0.0, 0.0, 0
    return total_recall / total_lumi, total_lumi, n


def compute_sigma_efficiency(
    *,
    mean_answer_metric: float,
    mean_recall_at_k: float,
) -> float | None:
    if mean_recall_at_k <= 1e-12:
        return None
    return float(mean_answer_metric) / float(mean_recall_at_k)


def cross_section_table(rows: Iterable[CrossSectionRow]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "method": r.method,
                "dataset": r.dataset,
                "k": r.k,
                "n_queries": r.n_queries,
                "luminosity": r.luminosity,
                "sigma_recall": r.sigma_recall,
                "sigma_efficiency": r.sigma_efficiency,
            }
        )
    return out
