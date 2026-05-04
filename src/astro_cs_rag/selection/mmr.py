"""Maximal Marginal Relevance (Carbonell & Goldstein 1998) — baseline diversity selector."""

from __future__ import annotations

import numpy as np

from astro_cs_rag.atoms.schemas import Chunk, EvidenceAtom
from astro_cs_rag.selection.greedy import SelectedRecord


def mmr_select(
    atoms: list[EvidenceAtom],
    chunks_by_id: dict[str, Chunk],
    embeddings_by_id: dict[str, np.ndarray],
    *,
    token_budget: int,
    lambda_param: float = 0.7,
) -> tuple[list[SelectedRecord], list[dict[str, object]], list[dict[str, object]]]:
    if not atoms:
        return [], [], []

    selected: list[SelectedRecord] = []
    dropped: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []

    # Per-query MMR. Without this grouping the algorithm becomes O(N^2) global
    # with semantically-bogus cross-query redundancy penalties.
    by_q: dict[str, list[EvidenceAtom]] = {}
    for a in atoms:
        by_q.setdefault(a.query_id, []).append(a)

    for qid, q_atoms in by_q.items():
        # Vectorize the redundancy penalty: cache embeddings once per query.
        cand_ids = [a.chunk_id for a in q_atoms]
        cand_snr = np.asarray([float(a.snr) for a in q_atoms], dtype=np.float32)
        emb_rows: list[np.ndarray | None] = [embeddings_by_id.get(cid) for cid in cand_ids]
        # Drop atoms with missing embeddings up-front.
        keep_idx = [i for i, v in enumerate(emb_rows) if v is not None]
        if not keep_idx:
            continue
        cand_ids = [cand_ids[i] for i in keep_idx]
        cand_snr = cand_snr[keep_idx]
        emb_mat = np.stack([emb_rows[i] for i in keep_idx], axis=0).astype(np.float32)

        n = len(cand_ids)
        selected_idx: list[int] = []
        # Running max similarity to any already-selected chunk.
        max_sim_to_selected = np.full(n, -np.inf, dtype=np.float32)
        used = 0

        while True:
            # Vectorized MMR score for unpicked chunks.
            avail_mask = np.ones(n, dtype=bool)
            for si in selected_idx:
                avail_mask[si] = False
            if not avail_mask.any():
                break
            penalty = np.where(np.isneginf(max_sim_to_selected), 0.0, max_sim_to_selected)
            score = lambda_param * cand_snr - (1.0 - lambda_param) * penalty
            score = np.where(avail_mask, score, -np.inf)
            i_best = int(np.argmax(score))
            if not np.isfinite(score[i_best]):
                break

            cid = cand_ids[i_best]
            ch = chunks_by_id.get(cid)
            if ch is None:
                dropped.append({"chunk_id": cid, "reason": "missing_chunk", "action": "drop", "query_id": qid})
                avail_mask[i_best] = False
                # mark consumed by setting score very low; loop again.
                max_sim_to_selected[i_best] = np.inf  # never reconsider
                continue
            if used + ch.token_count > token_budget:
                dropped.append({"chunk_id": ch.chunk_id, "reason": "budget", "action": "drop", "query_id": qid})
                # Budget can't fit this; try the next-best by retiring it.
                max_sim_to_selected[i_best] = np.inf
                continue

            selected.append(
                SelectedRecord(
                    query_id=qid, chunk_id=ch.chunk_id, snr=float(cand_snr[i_best]),
                    tokens=ch.token_count, reason=f"mmr_{lambda_param:.2f}",
                    metadata={"mmr_score": float(score[i_best])},
                )
            )
            trace.append({
                "query_id": qid, "chunk_id": ch.chunk_id, "action": "select",
                "mmr_score": float(score[i_best]),
                "cumulative_tokens": used + ch.token_count,
            })
            selected_idx.append(i_best)
            used += ch.token_count
            # Vectorized similarity update: cosine of all candidates to the just-picked one.
            sim_to_new = emb_mat @ emb_mat[i_best]
            max_sim_to_selected = np.maximum(max_sim_to_selected, sim_to_new)

    return selected, dropped, trace
