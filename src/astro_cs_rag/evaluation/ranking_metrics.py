"""Chunk-level precision and binary NDCG."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _chunk_relevant(
    chunk_id: str,
    chunk_to_doc: Mapping[str, str],
    gold_docs: set[str],
) -> bool:
    did = chunk_to_doc.get(chunk_id)
    return did is not None and did in gold_docs


def dcg_from_binary(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def precision_at_k_chunk(
    ranked_chunks: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_doc_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_doc_ids)
    if k <= 0 or not gold:
        return 0.0
    hits = sum(
        1
        for cid in ranked_chunks[:k]
        if _chunk_relevant(cid, chunk_to_doc, gold)
    )
    return hits / float(k)


def ndcg_at_k_chunk(
    ranked_chunks: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_doc_ids: list[str],
    k: int,
) -> float:
    gold = set(gold_doc_ids)
    if k <= 0 or not gold:
        return 0.0
    rels = [
        1.0 if _chunk_relevant(cid, chunk_to_doc, gold) else 0.0
        for cid in ranked_chunks[:k]
    ]
    ideal = sorted(rels, reverse=True)
    idcg = dcg_from_binary(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg_from_binary(rels) / idcg
