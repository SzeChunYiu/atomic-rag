"""Metric registry — doc-level recall/MRR + chunk precision/NDCG."""

from __future__ import annotations

from collections.abc import Mapping

from astro_cs_rag.atoms.schemas import Query
from astro_cs_rag.evaluation.ranking_metrics import ndcg_at_k_chunk, precision_at_k_chunk


def _doc_hits_in_prefix(
    ranked_chunks: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_docs: set[str],
    k: int,
) -> int:
    seen_docs: set[str] = set()
    for cid in ranked_chunks[:k]:
        did = chunk_to_doc.get(cid)
        if did is None:
            continue
        if did in gold_docs:
            seen_docs.add(did)
    return len(seen_docs & gold_docs)


def recall_at_k_doc(
    ranked_chunks: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_doc_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    hits = _doc_hits_in_prefix(ranked_chunks, chunk_to_doc, gold, k)
    return hits / len(gold)


def mrr_doc(
    ranked_chunks: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_doc_ids: list[str],
) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    for r, cid in enumerate(ranked_chunks, start=1):
        did = chunk_to_doc.get(cid)
        if did is not None and did in gold:
            return 1.0 / r
    return 0.0


def evaluate_ranked_queries(
    queries: list[Query],
    rankings: Mapping[str, list[str]],
    chunk_to_doc: Mapping[str, str],
    ks: list[int],
) -> dict[str, float]:
    """Aggregate metrics for registry-style logging."""
    metrics: dict[str, float] = {}
    for k in ks:
        recalls: list[float] = []
        precs: list[float] = []
        ndcgs: list[float] = []
        for q in queries:
            ranked = rankings.get(q.query_id, [])
            recalls.append(
                recall_at_k_doc(ranked, chunk_to_doc, q.gold_doc_ids, k)
            )
            precs.append(
                precision_at_k_chunk(ranked, chunk_to_doc, q.gold_doc_ids, k)
            )
            ndcgs.append(
                ndcg_at_k_chunk(ranked, chunk_to_doc, q.gold_doc_ids, k)
            )
        n = len(recalls)
        metrics[f"recall@{k}_doc_mean"] = sum(recalls) / n if n else 0.0
        metrics[f"precision@{k}_chunk_mean"] = sum(precs) / n if n else 0.0
        metrics[f"ndcg@{k}_chunk_mean"] = sum(ndcgs) / n if n else 0.0

    mrrs = []
    for q in queries:
        ranked = rankings.get(q.query_id, [])
        mrrs.append(mrr_doc(ranked, chunk_to_doc, q.gold_doc_ids))
    metrics["mean_reciprocal_rank_doc"] = sum(mrrs) / len(mrrs) if mrrs else 0.0
    return metrics
