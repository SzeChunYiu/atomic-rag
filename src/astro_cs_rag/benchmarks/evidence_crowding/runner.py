"""Run baselines on a crowding cell and produce result rows.

The runner is deliberately *generator-free*: it scores success by
oracle support-chain completeness, not by LLM output. This lets us
iterate on retrieval/selection diagnostics without paying generation
costs while we're still characterising the failure mode.

Two baselines are wired today:
  - ``atom_dense``: rank atoms by cos(query, atom), select top-budget.
  - ``chunk_dense``: rank chunks by cos(query, chunk_centroid), expand
                     to atoms in selected chunks.

Add deblending detector / graph reconstruction in `detection/` and
register them via :func:`register_system`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from astro_cs_rag.diagnostics.crowding_metrics import (
    all_gold_recall_at_k,
    support_chain_complete,
)
from astro_cs_rag.indexing.embedders import TrigramEmbedder

from .schema import CrowdingDataset, CrowdingResult


@dataclass
class Selection:
    selected_atom_ids: list[str]
    candidate_atom_ids: list[str]
    selected_tokens: int


SystemFn = Callable[[CrowdingDataset, str, "EmbeddingCache"], Selection]
_SYSTEMS: dict[str, SystemFn] = {}


def register_system(name: str, fn: SystemFn) -> None:
    _SYSTEMS[name] = fn


class EmbeddingCache:
    """Encode atoms/chunks/queries once per cell."""

    def __init__(self, embedder=None) -> None:
        self._emb = embedder or TrigramEmbedder()
        self.atom_emb: np.ndarray | None = None
        self.chunk_emb: np.ndarray | None = None
        self.atom_ids: list[str] = []
        self.chunk_ids: list[str] = []
        self.atom_idx: dict[str, int] = {}
        self.chunk_idx: dict[str, int] = {}
        self.query_emb: dict[str, np.ndarray] = {}

    def build(self, dataset: CrowdingDataset) -> None:
        self.atom_ids = [a.atom_id for a in dataset.atoms]
        self.chunk_ids = [c.chunk_id for c in dataset.chunks]
        self.atom_idx = {aid: i for i, aid in enumerate(self.atom_ids)}
        self.chunk_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.atom_emb = self._emb.encode([a.text for a in dataset.atoms])
        self.chunk_emb = self._emb.encode([c.text for c in dataset.chunks])
        for q in dataset.queries:
            self.query_emb[q.query_id] = self._emb.encode([q.text])[0]


def _topk_by_cos(query_vec: np.ndarray, mat: np.ndarray, ids: list[str], k: int) -> list[str]:
    if mat.size == 0 or k <= 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    scores = (mat @ q) / norms
    order = np.argsort(-scores)[:k]
    return [ids[i] for i in order]


def atom_dense(dataset: CrowdingDataset, query_id: str, cache: EmbeddingCache) -> Selection:
    q = next(qq for qq in dataset.queries if qq.query_id == query_id)
    qv = cache.query_emb[query_id]
    assert cache.atom_emb is not None
    cands = _topk_by_cos(qv, cache.atom_emb, cache.atom_ids, k=200)
    by_id = {a.atom_id: a for a in dataset.atoms}
    selected: list[str] = []
    tokens = 0
    for aid in cands:
        cost = max(1, len(by_id[aid].text.split()))
        if tokens + cost > q.text.count(" ") + dataset.cell.token_budget:
            break
        selected.append(aid)
        tokens += cost
    return Selection(selected_atom_ids=selected, candidate_atom_ids=cands, selected_tokens=tokens)


def chunk_dense(dataset: CrowdingDataset, query_id: str, cache: EmbeddingCache) -> Selection:
    q = next(qq for qq in dataset.queries if qq.query_id == query_id)
    qv = cache.query_emb[query_id]
    assert cache.chunk_emb is not None
    chunk_order = _topk_by_cos(qv, cache.chunk_emb, cache.chunk_ids, k=200)
    by_chunk = {c.chunk_id: c for c in dataset.chunks}
    by_id = {a.atom_id: a for a in dataset.atoms}
    selected: list[str] = []
    cand_atoms: list[str] = []
    tokens = 0
    for cid in chunk_order:
        for aid in by_chunk[cid].atom_ids:
            cand_atoms.append(aid)
            if aid in selected:
                continue
            cost = max(1, len(by_id[aid].text.split()))
            if tokens + cost > dataset.cell.token_budget:
                break
            selected.append(aid)
            tokens += cost
        if tokens >= dataset.cell.token_budget:
            break
    return Selection(selected_atom_ids=selected, candidate_atom_ids=cand_atoms, selected_tokens=tokens)


register_system("atom_dense", atom_dense)
register_system("chunk_dense", chunk_dense)


def _doc_recall(selected_atom_ids: list[str], gold_doc_ids: list[str], dataset: CrowdingDataset) -> float:
    by_atom = {a.atom_id: a for a in dataset.atoms}
    sel_docs = {by_atom[aid].doc_id for aid in selected_atom_ids if aid in by_atom}
    if not gold_doc_ids:
        return 0.0
    return len(set(gold_doc_ids) & sel_docs) / len(gold_doc_ids)


def run_cell(
    dataset: CrowdingDataset,
    systems: list[str],
    embedder=None,
) -> list[CrowdingResult]:
    cache = EmbeddingCache(embedder)
    cache.build(dataset)
    cell = dataset.cell
    rows: list[CrowdingResult] = []
    for sys_name in systems:
        fn = _SYSTEMS[sys_name]
        for q in dataset.queries:
            t0 = time.perf_counter()
            sel = fn(dataset, q.query_id, cache)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(
                CrowdingResult(
                    system_name=sys_name,
                    cell_id=cell.cell_id,
                    query_id=q.query_id,
                    n_distractors_per_gold=cell.n_distractors_per_gold,
                    semantic_similarity=cell.semantic_similarity,
                    entity_overlap=cell.entity_overlap,
                    chunk_size=cell.chunk_size,
                    hop_count=cell.hop_count,
                    token_budget=cell.token_budget,
                    gold_doc_recall_at_k=_doc_recall(sel.selected_atom_ids, q.gold_doc_ids, dataset),
                    gold_atom_recall_at_k=all_gold_recall_at_k(
                        sel.candidate_atom_ids, q.gold_atom_ids, k=200
                    ),
                    gold_atom_selected=all_gold_recall_at_k(
                        sel.selected_atom_ids, q.gold_atom_ids, k=len(sel.selected_atom_ids)
                    ),
                    support_chain_complete=support_chain_complete(
                        sel.selected_atom_ids, q.gold_atom_ids
                    ),
                    answer_oracle_success=support_chain_complete(
                        sel.selected_atom_ids, q.gold_atom_ids
                    ),
                    selected_tokens=sel.selected_tokens,
                    latency_ms=elapsed_ms,
                )
            )
    return rows
