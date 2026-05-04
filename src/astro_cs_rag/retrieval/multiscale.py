"""Multi-scale retrieval (RG-style: scale-dependent kernel).

Three scales:
- Scale D (coarse, doc-centroid): topic-level cosine similarity
- Scale C (medium, chunk): cosine + entity-overlap bonus
- Scale A (fine, atom): cosine + claim-type alignment

The non-trivial RG twist: kernel transforms with scale. Vanilla
hierarchical retrieval uses the same metric at all levels. Here:
- coarse: smooth topic kernel (cosine alone)
- medium: sharper kernel (cosine + entity overlap)
- fine: sharpest (cosine + entity overlap + claim-type alignment)

This addresses the bridge-recall failure mode: at coarse scale, both
gold docs co-cluster via the bridge entity; at fine scale, we extract
supporting facts. The metric flow ensures the right resolution at each
stage.
"""
from __future__ import annotations
import re
from collections import defaultdict
import numpy as np


_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


def query_entities(text: str) -> set[str]:
    """Proper-noun extraction; lowercase normalized."""
    return {m.group().lower() for m in _PROPER.finditer(text)}


def build_doc_centroids(
    chunk_embs: np.ndarray,        # (N_chunks, D), L2-normalized rows
    chunk_doc_ids: list[str],
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Aggregate chunk embeddings -> doc centroids.

    Returns (centroids, doc_id_list, chunk_idx_per_doc).
    Centroids are mean of normalized chunk embeddings, then re-normalized.
    """
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, did in enumerate(chunk_doc_ids):
        by_doc[did].append(i)
    doc_ids = sorted(by_doc.keys())
    cents = np.zeros((len(doc_ids), chunk_embs.shape[1]), dtype=np.float32)
    for k, did in enumerate(doc_ids):
        idx = by_doc[did]
        cents[k] = chunk_embs[idx].mean(axis=0)
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    return cents, doc_ids, dict(by_doc)


def multiscale_retrieve_atoms(
    *,
    query_emb: np.ndarray,         # (D,), L2-normalized
    query_text: str,
    atom_embs: np.ndarray,         # (N_atoms, D), L2-normalized
    atom_meta: list[dict],         # each {chunk_id, doc_id, claim_type, text}
    chunk_embs: np.ndarray,        # (N_chunks, D), L2-normalized
    chunk_meta: list[dict],        # each {chunk_id, doc_id, text}
    doc_centroids: np.ndarray,     # (N_docs, D)
    doc_ids: list[str],
    top_k_doc: int = 30,
    top_k_chunk: int = 80,
    top_k_atom: int = 50,
    entity_weight_chunk: float = 0.1,
    entity_weight_atom: float = 0.1,
    typed_weight_atom: float = 0.05,
    intent_type: str = "ANY",
    intent_conf: dict[str, float] | None = None,
) -> list[int]:
    """Returns indices into atom_embs/atom_meta of the top-k atoms
    selected via 3-scale RG-style retrieval."""
    qents = query_entities(query_text)

    # Scale D: doc centroids (smooth cosine)
    doc_scores = doc_centroids @ query_emb
    top_doc_idx = np.argpartition(-doc_scores, min(top_k_doc, len(doc_scores) - 1))[:top_k_doc]
    retained_docs = {doc_ids[i] for i in top_doc_idx}

    # Scale C: chunks within retained docs (cosine + entity overlap)
    chunk_in_doc = np.array(
        [c["doc_id"] in retained_docs for c in chunk_meta], dtype=bool)
    if not chunk_in_doc.any():
        return []
    chunk_idx = np.flatnonzero(chunk_in_doc)
    cs_scores = chunk_embs[chunk_idx] @ query_emb
    if entity_weight_chunk > 0 and qents:
        ent_bonus = np.array([
            sum(1 for e in qents if e in chunk_meta[i]["text"].lower())
            for i in chunk_idx], dtype=np.float32)
        cs_scores = cs_scores + entity_weight_chunk * ent_bonus
    k_c = min(top_k_chunk, len(cs_scores))
    top_chunk_local = np.argpartition(-cs_scores, k_c - 1)[:k_c]
    retained_chunks = {chunk_meta[chunk_idx[i]]["chunk_id"]
                       for i in top_chunk_local}

    # Scale A: atoms within retained chunks (cosine + entity + type)
    atom_in_chunk = np.array(
        [a["chunk_id"] in retained_chunks for a in atom_meta], dtype=bool)
    if not atom_in_chunk.any():
        return []
    atom_idx = np.flatnonzero(atom_in_chunk)
    as_scores = atom_embs[atom_idx] @ query_emb
    if entity_weight_atom > 0 and qents:
        ent_bonus = np.array([
            sum(1 for e in qents if e in atom_meta[i]["text"].lower())
            for i in atom_idx], dtype=np.float32)
        as_scores = as_scores + entity_weight_atom * ent_bonus
    if typed_weight_atom > 0:
        _SKIP = {"ANY", "WHAT_OBJ"}
        if intent_conf is not None:
            # Soft: dot-product of query type distribution × atom type distribution
            type_bonus = np.array([
                sum(
                    intent_conf.get(tp, 0.0) * atom_meta[i].get(
                        "claim_type_conf", {}).get(tp,
                        1.0 if atom_meta[i]["claim_type"] == tp else 0.0)
                    for tp in intent_conf if tp not in _SKIP
                )
                for i in atom_idx], dtype=np.float32)
            as_scores = as_scores + typed_weight_atom * type_bonus
        elif intent_type != "ANY":
            # Hard fallback (pre-built atoms without claim_type_conf)
            type_bonus = np.array([
                atom_meta[i].get("claim_type_conf", {}).get(intent_type,
                    1.0 if atom_meta[i]["claim_type"] == intent_type else 0.0)
                for i in atom_idx], dtype=np.float32)
            as_scores = as_scores + typed_weight_atom * type_bonus
    k_a = min(top_k_atom, len(as_scores))
    top_atom_local = np.argpartition(-as_scores, k_a - 1)[:k_a]
    sorted_local = top_atom_local[np.argsort(-as_scores[top_atom_local])]
    return [int(atom_idx[i]) for i in sorted_local]
