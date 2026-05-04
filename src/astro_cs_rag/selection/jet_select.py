"""Anti-$k_T$ jet selector adapter — bridges EvidenceAtom + chunk embeddings → SelectedRecord.

Three modes governed by ``n_jets``:
  -  ``n_jets >= 1``: pack only the top-N jets in jet-relevance order
     (members sorted by SNR desc within each jet). Aggressive: leading jet
     exclusion of non-leading-jet chunks. v1 default; works on synthetic
     joint-evidence stress tests, collapses on multi-hop QA.
  -  ``n_jets == -1``: pack across ALL jets in jet-relevance order, members
     sorted by SNR. Effectively reduces to greedy-by-SNR when the leading
     jet's high-SNR member already dominates the budget.
  -  ``n_jets == -2``: ``v3 atomic-unit greedy``. Greedy pack chunks by SNR
     desc; when a chunk is selected, also include its highest-SNR jet
     partner (one pull-in per selection) regardless of partner SNR rank.
     Preserves greedy on ranked-evidence queries; recovers joint-evidence
     pairs that anti-kT clusters together. The IRC-safety mechanism is
     preserved (clustering still uses anti-kT distances) but it acts as
     a *soft preference* on top of greedy rather than a *hard exclusion*.
"""

from __future__ import annotations

import numpy as np

from astro_cs_rag.atoms.schemas import Chunk, EvidenceAtom
from astro_cs_rag.selection.anti_kt import cluster_anti_kt
from astro_cs_rag.selection.greedy import SelectedRecord


def _select_v3_atomic_unit(
    qid: str,
    ids: list[str],
    snr_lookup: dict[str, float],
    chunks_by_id: dict[str, Chunk],
    jets,
    *,
    token_budget: int,
    R: float,
    partner_gate_alpha: float,
    partner_use_median: bool,
    selected: list[SelectedRecord],
    dropped: list[dict[str, object]],
    trace: list[dict[str, object]],
) -> None:
    chunk_to_jet: dict[str, int] = {}
    jet_members: dict[int, list[str]] = {}
    for ji, jet in enumerate(jets):
        jet_members[ji] = list(jet.member_atom_ids)
        for cid in jet.member_atom_ids:
            chunk_to_jet[cid] = ji
    chunks_by_snr = sorted(ids, key=lambda c: -float(snr_lookup.get(c, 0.0)))
    seen: set[str] = set()
    used = 0
    # v4 partner gate: precompute median floor (over candidate SNRs for this query).
    median_floor = 0.0
    if partner_use_median and ids:
        snr_arr = np.fromiter((float(snr_lookup.get(c, 0.0)) for c in ids), dtype=np.float64, count=len(ids))
        median_floor = float(np.median(snr_arr))
    for cid in chunks_by_snr:
        if cid in seen:
            continue
        ch = chunks_by_id.get(cid)
        if ch is None:
            dropped.append({"query_id": qid, "chunk_id": cid, "reason": "missing_chunk", "action": "drop"})
            continue
        if used + ch.token_count > token_budget:
            dropped.append({"query_id": qid, "chunk_id": cid, "reason": "budget", "action": "drop"})
            continue
        # Select the chunk itself.
        selected.append(SelectedRecord(
            query_id=qid, chunk_id=cid, snr=float(snr_lookup.get(cid, 0.0)),
            tokens=ch.token_count, reason=f"anti_kt_v3_R{R:.2f}",
            metadata={"jet_index": int(chunk_to_jet.get(cid, -1)), "role": "primary"},
        ))
        trace.append({"query_id": qid, "chunk_id": cid, "action": "select", "role": "primary",
                      "cumulative_tokens": used + ch.token_count})
        seen.add(cid)
        used += ch.token_count
        # Atomic-unit pull-in: ALL jet partners not yet seen, in merge order.
        # Anti-kT merge order (which chunks fused first under d_ij minimum)
        # encodes joint-evidence affinity; SNR-sort breaks gold-pair atomicity.
        ji = chunk_to_jet.get(cid)
        if ji is None:
            continue
        partners = [m for m in jet_members[ji] if m != cid and m not in seen]
        primary_snr = float(snr_lookup.get(cid, 0.0))
        gate = max(primary_snr * partner_gate_alpha, median_floor) if (partner_gate_alpha > 0.0 or partner_use_median) else float("-inf")
        for partner in partners:
            pch = chunks_by_id.get(partner)
            if pch is None:
                continue
            partner_snr = float(snr_lookup.get(partner, 0.0))
            if partner_snr < gate:
                dropped.append({"query_id": qid, "chunk_id": partner, "reason": "partner_score_gate", "action": "drop"})
                continue
            if used + pch.token_count > token_budget:
                dropped.append({"query_id": qid, "chunk_id": partner, "reason": "budget_partner", "action": "drop"})
                continue
            tag = "anti_kt_v4" if (partner_gate_alpha > 0.0 or partner_use_median) else "anti_kt_v3"
            selected.append(SelectedRecord(
                query_id=qid, chunk_id=partner, snr=partner_snr,
                tokens=pch.token_count, reason=f"{tag}_R{R:.2f}",
                metadata={"jet_index": int(ji), "role": "jet_partner_of", "primary_chunk_id": cid},
            ))
            trace.append({"query_id": qid, "chunk_id": partner, "action": "select",
                          "role": "jet_partner", "primary_chunk_id": cid,
                          "cumulative_tokens": used + pch.token_count})
            seen.add(partner)
            used += pch.token_count


def jet_select(
    atoms: list[EvidenceAtom],
    chunks_by_id: dict[str, Chunk],
    embeddings_by_id: dict[str, np.ndarray],
    *,
    token_budget: int,
    R: float = 1.0,
    n_jets: int = 1,
    partner_score_gate_alpha: float = 0.0,
    partner_use_median_floor: bool = False,
) -> tuple[list[SelectedRecord], list[dict[str, object]], list[dict[str, object]]]:
    if not atoms:
        return [], [], []

    by_q: dict[str, list[EvidenceAtom]] = {}
    for a in atoms:
        by_q.setdefault(a.query_id, []).append(a)

    selected: list[SelectedRecord] = []
    dropped: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []

    for qid, parts in by_q.items():
        ids: list[str] = []
        relevances: list[float] = []
        emb_rows: list[np.ndarray] = []
        for a in parts:
            v = embeddings_by_id.get(a.chunk_id)
            if v is None:
                continue
            ids.append(a.chunk_id)
            relevances.append(float(a.snr))
            emb_rows.append(v)
        if not ids:
            continue
        emb_mat = np.stack(emb_rows, axis=0).astype(np.float32)
        result = cluster_anti_kt(ids, relevances, emb_mat, R=R)

        snr_lookup = {a.chunk_id: a.snr for a in parts}
        if n_jets == -2:
            # v3 atomic-unit greedy: greedy-by-SNR with one jet-partner pull-in.
            _select_v3_atomic_unit(
                qid, ids, snr_lookup, chunks_by_id, result.final_jets,
                token_budget=token_budget, R=R,
                partner_gate_alpha=partner_score_gate_alpha,
                partner_use_median=partner_use_median_floor,
                selected=selected, dropped=dropped, trace=trace,
            )
            continue

        # Pack across the top n_jets in jet-relevance order. Two member orderings:
        #  - n_jets >= 1 (aggressive leading-jet mode): KEEP merge order — gold-pair
        #    atomicity depends on members entering the budget in the order anti-kT
        #    clustered them. Sorting by SNR breaks the joint-evidence guarantee.
        #  - n_jets == -1 (soft IRC ranker): sort members by SNR — high-relevance
        #    chunks first so this approximates greedy when the leading jet
        #    dominates. (Used as the safe-default mode.)
        if n_jets < 0:
            jets_to_pack = result.final_jets
        else:
            jets_to_pack = result.final_jets[:n_jets]
        ranked_chunks: list[tuple[str, float]] = []
        for jet in jets_to_pack:
            if n_jets < 0:
                members = sorted(jet.member_atom_ids, key=lambda c: -float(snr_lookup.get(c, 0.0)))
            else:
                members = list(jet.member_atom_ids)  # internal merge order
            for cid in members:
                ranked_chunks.append((cid, jet.relevance))

        seen: set[str] = set()
        used = 0
        for cid, rel in ranked_chunks:
            if cid in seen:
                continue
            ch = chunks_by_id.get(cid)
            if ch is None:
                dropped.append({"query_id": qid, "chunk_id": cid, "reason": "missing_chunk", "action": "drop"})
                continue
            if used + ch.token_count > token_budget:
                dropped.append({"query_id": qid, "chunk_id": cid, "reason": "budget", "action": "drop"})
                continue
            selected.append(
                SelectedRecord(
                    query_id=qid,
                    chunk_id=ch.chunk_id,
                    snr=float(snr_lookup.get(cid, 0.0)),
                    tokens=ch.token_count,
                    reason=f"anti_kt_R{R:.2f}",
                    metadata={"jet_relevance": float(rel)},
                )
            )
            trace.append(
                {
                    "query_id": qid,
                    "chunk_id": ch.chunk_id,
                    "action": "select",
                    "jet_relevance": float(rel),
                    "cumulative_tokens": used + ch.token_count,
                }
            )
            seen.add(cid)
            used += ch.token_count

    return selected, dropped, trace
