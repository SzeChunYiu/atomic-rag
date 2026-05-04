"""Budgeted greedy selection ordered by detector SNR."""

from __future__ import annotations

from pydantic import BaseModel, Field

from astro_cs_rag.atoms.schemas import Chunk, EvidenceAtom
from astro_cs_rag.selection.coverage import token_overlap_fraction


class SelectedRecord(BaseModel):
    query_id: str
    chunk_id: str
    snr: float
    tokens: int
    reason: str = "snr_order"
    metadata: dict[str, object] = Field(default_factory=dict)


def greedy_select(
    atoms: list[EvidenceAtom],
    chunks_by_id: dict[str, Chunk],
    *,
    token_budget: int,
    query_text: str | None = None,
) -> tuple[list[SelectedRecord], list[dict[str, object]], list[dict[str, object]]]:
    """Return selected rows, dropped rows, and coverage trace rows."""
    ordered = sorted(atoms, key=lambda a: a.snr, reverse=True)
    selected: list[SelectedRecord] = []
    dropped: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    used = 0
    seen: set[str] = set()
    for atom in ordered:
        ch = chunks_by_id.get(atom.chunk_id)
        overlap = (
            token_overlap_fraction(query_text, ch.text)
            if query_text is not None and ch is not None
            else 0.0
        )
        if ch is None:
            row = {
                "chunk_id": atom.chunk_id,
                "reason": "missing_chunk",
                "overlap_fraction": 0.0,
                "action": "drop",
            }
            dropped.append(row)
            trace.append({**row, "query_id": atom.query_id})
            continue
        if ch.chunk_id in seen:
            row = {"chunk_id": ch.chunk_id, "reason": "duplicate", "overlap_fraction": overlap, "action": "drop"}
            dropped.append(row)
            trace.append({**row, "query_id": atom.query_id})
            continue
        if used + ch.token_count > token_budget:
            row = {
                "chunk_id": ch.chunk_id,
                "reason": "budget",
                "overlap_fraction": overlap,
                "action": "drop",
            }
            dropped.append(row)
            trace.append({**row, "query_id": atom.query_id})
            continue
        used_after = used + ch.token_count
        selected.append(
            SelectedRecord(
                query_id=atom.query_id,
                chunk_id=ch.chunk_id,
                snr=atom.snr,
                tokens=ch.token_count,
                metadata={"overlap_fraction": overlap},
            )
        )
        trace.append(
            {
                "query_id": atom.query_id,
                "chunk_id": ch.chunk_id,
                "action": "select",
                "overlap_fraction": overlap,
                "cumulative_tokens": used_after,
                "snr": atom.snr,
            }
        )
        used = used_after
        seen.add(ch.chunk_id)
    return selected, dropped, trace
