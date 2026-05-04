"""Turn fused retrieval scores into typed Candidate rows."""

from __future__ import annotations

from astro_cs_rag.atoms.schemas import Candidate


def build_candidates(
    query_id: str,
    fused_scores: dict[str, float],
    *,
    top_n: int,
    retriever_label: str = "fusion_rrf",
) -> list[Candidate]:
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[
        :max(0, top_n)
    ]
    out: list[Candidate] = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        out.append(
            Candidate(
                query_id=query_id,
                chunk_id=chunk_id,
                raw_score=float(score),
                retriever=retriever_label,
                rank=rank,
                metadata={},
            )
        )
    return out
