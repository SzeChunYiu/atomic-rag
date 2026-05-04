"""Merge ranked lists — RRF is robust when BM25 and dense disagree."""

from __future__ import annotations

from collections import defaultdict


def rank_by_score(scores: dict[str, float]) -> list[str]:
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranks in ranked_lists:
        for rank, cid in enumerate(ranks, start=1):
            fused[cid] += 1.0 / (k + rank)
    return dict(fused)
