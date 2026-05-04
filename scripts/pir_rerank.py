"""Apply Path-Integral Retrieval reranking to a post-retrieve run dir.

Reads:
  - <run_dir>/candidates.jsonl       (top-50 from dense retrieval)
  - <run_dir>/../index_bundle/embeddings.npy + chunks.jsonl
  - <run_dir>/config.yaml            (gives queries path)

Writes:
  - <run_dir>/candidates.jsonl       (rewritten in-place)
  - <run_dir>/pir_meta.json          (timing, parameters)

Then the rest of the pipeline (selector + generator) consumes the
PIR-reordered candidates with reranker.enabled=False.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")
from astro_cs_rag.reranking.path_integral import path_integral_rerank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top_n_in", type=int, default=50)
    ap.add_argument("--top_n_out", type=int, default=20)
    ap.add_argument("--edge_threshold", type=float, default=0.5)
    ap.add_argument("--max_path_length", type=int, default=3)
    ap.add_argument("--direct_weight", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = args.run_dir
    parent = run_dir.parent  # for 2wiki/realgen layouts where index_bundle is sibling of run dirs
    idx_dir = parent / "index_bundle"
    if not idx_dir.is_dir():
        # fallback: index inside run_dir
        idx_dir = run_dir / "index_bundle"

    # load embeddings and chunk_ids
    embs = np.load(idx_dir / "embeddings.npy")
    # normalize to unit
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = (embs / norms).astype(np.float32)
    chunk_ids = []
    for line in open(idx_dir / "chunks.jsonl"):
        c = json.loads(line)
        chunk_ids.append(c["chunk_id"])
    cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    # load query embeddings — re-encode from config queries
    cfg_raw = yaml.safe_load(open(run_dir / "config.yaml"))
    queries_path = Path(cfg_raw["paths"]["queries_path"])
    if not queries_path.is_absolute():
        queries_path = Path("/projects/hep/fs10/shared/nnbar/billy/RAG") / queries_path
    queries: dict[str, str] = {}
    for line in open(queries_path):
        q = json.loads(line)
        queries[q["query_id"]] = q["text"]

    # encode queries with same embedder as index
    from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
    from astro_cs_rag.config.schema import EmbeddingSettings
    _, _, _, meta = load_index_bundle(idx_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())
    qids = list(queries.keys())
    q_embs = embedder.encode([queries[q] for q in qids])
    q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9
    q_embs = (q_embs / q_norms).astype(np.float32)
    qid_to_emb = {qid: q_embs[i] for i, qid in enumerate(qids)}

    # group candidates by query
    by_q: dict[str, list[dict]] = {}
    for line in open(run_dir / "candidates.jsonl"):
        c = json.loads(line)
        by_q.setdefault(c["query_id"], []).append(c)

    # apply PIR per query
    new_records = []
    t0 = time.perf_counter()
    for qid, parts in by_q.items():
        parts.sort(key=lambda x: -x["raw_score"])
        in_pool = parts[: args.top_n_in]
        local_chunk_ids = [p["chunk_id"] for p in in_pool]
        # local embedding pool
        local_idx = [cid_to_idx[c] for c in local_chunk_ids if c in cid_to_idx]
        if len(local_idx) < 2:
            new_records.extend(in_pool[: args.top_n_out])
            continue
        local_embs = embs[local_idx]
        valid_ids = [chunk_ids[i] for i in local_idx]
        q_emb = qid_to_emb.get(qid)
        if q_emb is None:
            new_records.extend(in_pool[: args.top_n_out])
            continue
        ranked = path_integral_rerank(
            q_emb,
            local_embs,
            valid_ids,
            edge_threshold=args.edge_threshold,
            max_path_length=args.max_path_length,
            direct_weight=args.direct_weight,
        )
        # rewrite candidates with new scores + ranks
        score_by_id = {cid: sc for cid, sc in ranked}
        out_pool = sorted(in_pool, key=lambda x: -score_by_id.get(x["chunk_id"], -1e9))[: args.top_n_out]
        for rank, c in enumerate(out_pool, start=1):
            c2 = dict(c)
            c2["raw_score"] = float(score_by_id.get(c["chunk_id"], 0.0))
            c2["retriever"] = c.get("retriever", "dense") + "+pir"
            c2["rank"] = rank
            md = dict(c.get("metadata", {}))
            md["pre_pir_score"] = c["raw_score"]
            md["pre_pir_rank"] = c.get("rank")
            c2["metadata"] = md
            new_records.append(c2)
    elapsed = time.perf_counter() - t0

    # rewrite candidates.jsonl
    with open(run_dir / "candidates.jsonl", "w") as f:
        for c in new_records:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(run_dir / "pir_meta.json", "w") as f:
        json.dump({
            "edge_threshold": args.edge_threshold,
            "max_path_length": args.max_path_length,
            "direct_weight": args.direct_weight,
            "top_n_in": args.top_n_in,
            "top_n_out": args.top_n_out,
            "elapsed_seconds": elapsed,
            "n_queries": len(by_q),
        }, f, indent=2)
    print(f"PIR: {len(by_q)} queries, {elapsed:.2f}s ({elapsed/max(1,len(by_q))*1000:.2f}ms/q)")


if __name__ == "__main__":
    main()
