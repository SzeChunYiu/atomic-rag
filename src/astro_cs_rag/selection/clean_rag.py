"""CLEAN-RAG: radio-astronomy CLEAN deconvolution adapted to evidence selection.

Högbom CLEAN (1974) iteratively picks the brightest pixel and subtracts a
scaled point-spread function from the residual. We adapt:

  1. Pick the candidate atom with highest *residual* relevance.
  2. Subtract its semantic projection (gain * its embedding) from the
     residual *query need vector*.
  3. Repeat until the residual norm falls below a threshold or a maximum
     number of iterations / token budget is reached.

The residual query need vector starts as the original query embedding and
shrinks as covered information is "explained away." This is the
inverse-problem complement to anti-$k_T$ clustering: instead of grouping
atoms into jets, we sequentially explain the query with one atom at a time.
"""

from __future__ import annotations

import numpy as np

from astro_cs_rag.atoms.schemas import Chunk, EvidenceAtom
from astro_cs_rag.selection.greedy import SelectedRecord


def clean_select(
    *,
    atoms: list[EvidenceAtom],
    chunks_by_id: dict[str, Chunk],
    embeddings_by_id: dict[str, np.ndarray],
    query_embeddings: dict[str, np.ndarray],
    token_budget: int,
    gain: float = 0.7,
    residual_floor: float = 0.05,
    max_iters: int = 20,
) -> tuple[list[SelectedRecord], list[dict[str, object]], list[dict[str, object]]]:
    """Iteratively select atoms that reduce the per-query residual.

    `gain` is the CLEAN gain factor (0 < gain ≤ 1). Smaller gain → more,
    smaller iterations (better deconvolution, slower).
    `query_embeddings` is a per-query map; queries without an embedding are
    skipped.
    """
    if not atoms:
        return [], [], []

    by_q: dict[str, list[EvidenceAtom]] = {}
    for a in atoms:
        by_q.setdefault(a.query_id, []).append(a)

    selected: list[SelectedRecord] = []
    dropped: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []

    for qid, parts in by_q.items():
        q_emb = query_embeddings.get(qid)
        if q_emb is None:
            continue
        residual = q_emb.astype(np.float32).copy()
        residual /= np.linalg.norm(residual) + 1e-9
        used = 0
        seen: set[str] = set()
        for it in range(max_iters):
            if np.linalg.norm(residual) <= residual_floor:
                break
            best_inner = -float("inf")
            best_atom: EvidenceAtom | None = None
            best_emb = None
            for a in parts:
                if a.chunk_id in seen:
                    continue
                v = embeddings_by_id.get(a.chunk_id)
                if v is None:
                    continue
                v = v / (np.linalg.norm(v) + 1e-9)
                inner = float(np.dot(residual, v.astype(np.float32))) * float(a.snr)
                if inner > best_inner:
                    best_inner = inner
                    best_atom = a
                    best_emb = v
            if best_atom is None or best_emb is None or best_inner <= 0:
                break
            ch = chunks_by_id.get(best_atom.chunk_id)
            if ch is None:
                dropped.append({"query_id": qid, "chunk_id": best_atom.chunk_id, "reason": "missing_chunk", "action": "drop"})
                seen.add(best_atom.chunk_id)
                continue
            if used + ch.token_count > token_budget:
                dropped.append({"query_id": qid, "chunk_id": best_atom.chunk_id, "reason": "budget", "action": "drop"})
                seen.add(best_atom.chunk_id)
                continue
            selected.append(
                SelectedRecord(
                    query_id=qid,
                    chunk_id=ch.chunk_id,
                    snr=float(best_atom.snr),
                    tokens=ch.token_count,
                    reason=f"clean_g{gain:.2f}",
                    metadata={"clean_iter": it, "clean_inner": best_inner},
                )
            )
            trace.append(
                {
                    "query_id": qid,
                    "chunk_id": ch.chunk_id,
                    "action": "select",
                    "iter": it,
                    "residual_norm": float(np.linalg.norm(residual)),
                    "clean_inner": best_inner,
                }
            )
            seen.add(ch.chunk_id)
            used += ch.token_count
            # Subtract gain * inner * direction from residual.
            residual = residual - float(gain) * best_inner * best_emb.astype(np.float32)

    return selected, dropped, trace
