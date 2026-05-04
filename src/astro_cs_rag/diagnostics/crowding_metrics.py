"""Evidence-crowding metrics on atom catalogs.

Crowding here means: how many *non-gold* atoms look similar to a gold
atom under one of several similarity spaces (dense embedding, claim
type, entity overlap, carrier chunk, query-conditioned path).

These are diagnostic measurements, not selectors. They feed the phase
diagram (`phase_transition.py`) and the audit report.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dense_crowding(
    gold_atom_idx: int,
    atom_embs: np.ndarray,
    gold_atom_idxs: set[int],
    radius: float,
) -> int:
    """Count non-gold atoms within cosine distance `radius` of the gold atom.

    Atoms are L2-normalised internally, so `radius` is interpretable as
    `1 - cos`. Cosines ≥ (1 - radius) qualify as neighbors.
    """
    if atom_embs.size == 0 or gold_atom_idx >= len(atom_embs):
        return 0
    norms = np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-12
    unit = atom_embs / norms
    g = unit[gold_atom_idx]
    cos = unit @ g
    threshold = 1.0 - radius
    mask = cos >= threshold
    mask[gold_atom_idx] = False
    for j in gold_atom_idxs:
        if j < len(mask):
            mask[j] = False
    return int(mask.sum())


def type_crowding(
    gold_claim_type: str,
    candidate_claim_types: Sequence[str],
    gold_idxs_in_candidates: set[int],
) -> int:
    """Count candidate atoms with the same claim type as the gold atom,
    excluding the gold atoms themselves."""
    return sum(
        1
        for i, t in enumerate(candidate_claim_types)
        if t == gold_claim_type and i not in gold_idxs_in_candidates
    )


def entity_crowding(
    query_entities: set[str],
    gold_entities: set[str],
    candidate_entities: Sequence[set[str]],
    gold_idxs_in_candidates: set[int],
) -> int:
    """Count candidates sharing at least one entity with the query or gold,
    excluding gold atoms. Empty entity sets do not count."""
    target = query_entities | gold_entities
    if not target:
        return 0
    return sum(
        1
        for i, ents in enumerate(candidate_entities)
        if i not in gold_idxs_in_candidates and (ents & target)
    )


def chunk_crowding(
    gold_chunk_id: str,
    atom_chunk_ids: Sequence[str],
    gold_atom_idxs: set[int],
) -> int:
    """Count non-gold atoms inside the gold carrier chunk."""
    return sum(
        1
        for i, cid in enumerate(atom_chunk_ids)
        if cid == gold_chunk_id and i not in gold_atom_idxs
    )


def support_chain_complete(
    selected_atom_ids: Sequence[str],
    required_gold_atom_ids: Sequence[str],
) -> bool:
    """All required gold atoms must appear in the selected set."""
    if not required_gold_atom_ids:
        return False
    sel = set(selected_atom_ids)
    return all(g in sel for g in required_gold_atom_ids)


def all_gold_recall_at_k(
    candidate_atom_ids: Sequence[str],
    required_gold_atom_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of required gold atoms that appear in top-k candidates."""
    if not required_gold_atom_ids:
        return 0.0
    topk = set(candidate_atom_ids[:k])
    hit = sum(1 for g in required_gold_atom_ids if g in topk)
    return hit / len(required_gold_atom_ids)
