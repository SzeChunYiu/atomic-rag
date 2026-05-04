"""Lexical overlap heuristic for coverage (placeholder for facet coverage)."""

from __future__ import annotations


def token_overlap_fraction(query: str, chunk: str) -> float:
    qt = set(query.lower().split())
    ct = set(chunk.lower().split())
    if not qt:
        return 0.0
    return len(qt & ct) / len(qt)
