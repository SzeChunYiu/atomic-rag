"""Hierarchical (RAPTOR-style) chunk tree.

Original RAPTOR (Sarthi et al. 2024) uses UMAP + GMM clustering with
LLM-generated summaries. We implement a deterministic, compute-cheap variant:

  level 0:    leaf chunks
  level ℓ+1:  agglomerative clustering on level-ℓ embeddings; each cluster's
              representative text is the concatenation of member texts (kept
              under a token cap), and its embedding is the mean of member
              embeddings (re-normalized).

This avoids LLM calls in the *baseline*; in P5+ we can switch summaries to
LLM-generated and still keep this path as a falsifiable comparison ("does
the LLM summary actually help?").

Retrieval over the tree: collapsed top-k across *all* levels. We do not
implement top-down hierarchical descent in the baseline because the
collapsed variant is the strongest reported in the original paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class HierarchicalNode:
    node_id: str
    level: int
    text: str
    member_chunk_ids: list[str]   # original (level-0) chunk ids covered by this node
    embedding: np.ndarray         # (d,) float32, L2-normalized


def _agglomerative_pairs(emb: np.ndarray, n_clusters: int) -> list[list[int]]:
    """Cosine-distance agglomerative clustering, average linkage."""
    from sklearn.cluster import AgglomerativeClustering

    if emb.shape[0] <= n_clusters:
        return [[i] for i in range(emb.shape[0])]
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(emb)
    out: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(int(lab), []).append(i)
    return [out[k] for k in sorted(out)]


def _summary_text(member_texts: Sequence[str], char_cap: int = 800) -> str:
    """Deterministic concatenation under a character cap (no LLM)."""
    text = " // ".join(t.strip() for t in member_texts if t.strip())
    if len(text) <= char_cap:
        return text
    # Round-robin truncation: keep a prefix of each member proportional to length.
    parts = [t.strip() for t in member_texts if t.strip()]
    if not parts:
        return ""
    per = max(50, char_cap // len(parts))
    return " // ".join(p[:per] for p in parts)[:char_cap]


def build_hierarchy(
    leaf_chunk_ids: list[str],
    leaf_texts: list[str],
    leaf_embeddings: np.ndarray,
    *,
    branching: int = 6,
    max_levels: int = 4,
    summary_char_cap: int = 800,
) -> list[HierarchicalNode]:
    """Build levels [0, 1, ..., L] with branching factor `branching`.

    Returns a flat list of nodes (level-0 leaves included) in build order.
    """
    if len(leaf_chunk_ids) != len(leaf_texts) or len(leaf_chunk_ids) != leaf_embeddings.shape[0]:
        msg = "leaf inputs length mismatch"
        raise ValueError(msg)
    if leaf_embeddings.size == 0:
        return []

    nodes: list[HierarchicalNode] = []
    # Level 0 — leaves.
    for cid, txt, emb in zip(leaf_chunk_ids, leaf_texts, leaf_embeddings, strict=True):
        nodes.append(
            HierarchicalNode(
                node_id=f"{cid}",
                level=0,
                text=txt,
                member_chunk_ids=[cid],
                embedding=emb.astype(np.float32),
            )
        )

    current_indices = list(range(len(nodes)))
    level = 0
    while True:
        if len(current_indices) <= 1:
            break
        if level >= max_levels:
            break
        n_clusters = max(1, len(current_indices) // max(2, branching))
        if n_clusters >= len(current_indices):
            break

        cur_emb = np.stack([nodes[i].embedding for i in current_indices], axis=0)
        clusters = _agglomerative_pairs(cur_emb, n_clusters=n_clusters)

        next_indices: list[int] = []
        for cl_idx, member_local_idx in enumerate(clusters):
            members = [nodes[current_indices[k]] for k in member_local_idx]
            texts = [m.text for m in members]
            mean = np.mean(np.stack([m.embedding for m in members], axis=0), axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            child_chunk_ids: list[str] = []
            for m in members:
                child_chunk_ids.extend(m.member_chunk_ids)
            new_id = f"L{level + 1}_c{cl_idx}_{len(nodes)}"
            new_node = HierarchicalNode(
                node_id=new_id,
                level=level + 1,
                text=_summary_text(texts, char_cap=summary_char_cap),
                member_chunk_ids=child_chunk_ids,
                embedding=mean.astype(np.float32),
            )
            nodes.append(new_node)
            next_indices.append(len(nodes) - 1)
        current_indices = next_indices
        level += 1

    return nodes


def hierarchy_scores(
    nodes: list[HierarchicalNode],
    query_embedding: np.ndarray,
    *,
    flatten_to_chunks: bool = True,
) -> dict[str, float]:
    """Cosine score of query vs every node; collapse to underlying leaf chunk ids.

    When `flatten_to_chunks=True`, each node propagates its score to its
    constituent leaves (max over node-paths). This implements the "collapsed
    tree" retrieval variant.
    """
    if not nodes:
        return {}
    q = query_embedding.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    # Vectorize: single matmul over all node embeddings (already L2-normalized
    # per HierarchicalNode invariant). Previously this was a Python loop with
    # one np.dot per node — O(N) Python-level overhead per query.
    emb_mat = np.stack([n.embedding for n in nodes], axis=0).astype(np.float32)
    sims = (emb_mat @ q).tolist()
    out: dict[str, float] = {}
    if flatten_to_chunks:
        for n, score in zip(nodes, sims, strict=True):
            for cid in n.member_chunk_ids:
                if cid not in out or score > out[cid]:
                    out[cid] = score
    else:
        for n, score in zip(nodes, sims, strict=True):
            out[n.node_id] = score
    return out


def save_hierarchy(nodes: list[HierarchicalNode], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not nodes:
        np.save(directory / "raptor_emb.npy", np.zeros((0, 0), dtype=np.float32))
        (directory / "raptor_meta.jsonl").write_text("", encoding="utf-8")
        return
    emb = np.stack([n.embedding for n in nodes], axis=0).astype(np.float32)
    np.save(directory / "raptor_emb.npy", emb)
    rows = [
        {
            "node_id": n.node_id,
            "level": n.level,
            "text": n.text,
            "members": n.member_chunk_ids,
        }
        for n in nodes
    ]
    (directory / "raptor_meta.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def load_hierarchy(directory: Path) -> list[HierarchicalNode]:
    emb = np.load(directory / "raptor_emb.npy")
    out: list[HierarchicalNode] = []
    if not (directory / "raptor_meta.jsonl").is_file():
        return out
    for i, line in enumerate((directory / "raptor_meta.jsonl").read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append(
            HierarchicalNode(
                node_id=str(r["node_id"]),
                level=int(r["level"]),
                text=str(r["text"]),
                member_chunk_ids=list(r["members"]),
                embedding=emb[i].astype(np.float32),
            )
        )
    return out
