"""Graph-Walk Retrieval (GWR) — 2-hop expansion of dense top-K.

For each query:
  1. s = cos(q, all_chunks); take top-K1 by s (primary set).
  2. Expand: for each chunk in primary, find its top-K2 neighbors by cos.
  3. Union: expanded candidate set.
  4. Score each chunk in expanded set as:
       score[i] = max(s[i], max_k {s[k] * cos(k, i)} for k in primary)
     (max-product propagation through 1 hop)
  5. Output top-50 by score.

Replaces candidates.jsonl in a post-retrieve run dir.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top_k1", type=int, default=20, help="primary chunks for hop-1")
    ap.add_argument("--top_k2", type=int, default=10, help="neighbors per primary")
    ap.add_argument("--top_n_out", type=int, default=50)
    args = ap.parse_args()

    rd = args.run_dir
    parent = rd.parent
    idx = parent / "index_bundle"

    embs = np.load(idx / "embeddings.npy")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = (embs / norms).astype(np.float32)

    chunk_ids = []
    for line in open(idx / "chunks.jsonl"):
        c = json.loads(line); chunk_ids.append(c["chunk_id"])
    cid_to_idx = {c: i for i, c in enumerate(chunk_ids)}

    # Load queries text + encode
    cfg = yaml.safe_load(open(rd / "config.yaml"))
    qpath = Path(cfg["paths"]["queries_path"])
    if not qpath.is_absolute():
        qpath = Path("/projects/hep/fs10/shared/nnbar/billy/RAG") / qpath
    queries = {}
    for line in open(qpath):
        q = json.loads(line)
        queries[q["query_id"]] = q["text"]

    from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
    from astro_cs_rag.config.schema import EmbeddingSettings
    _, _, _, meta = load_index_bundle(idx)
    embedder = embedder_from_meta(meta, EmbeddingSettings())
    qids = list(queries.keys())
    q_embs = embedder.encode([queries[q] for q in qids])
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    q_embs = q_embs.astype(np.float32)

    # Read existing candidate.jsonl to preserve metadata for chosen chunks
    by_q: dict[str, list[dict]] = {}
    for line in open(rd / "candidates.jsonl"):
        c = json.loads(line)
        by_q.setdefault(c["query_id"], []).append(c)

    new_records = []
    t0 = time.time()
    n_q = 0
    for qi, qid in enumerate(qids):
        q_emb = q_embs[qi]
        s = embs @ q_emb  # (N,)
        top_k1_idx = np.argpartition(-s, args.top_k1)[:args.top_k1]
        # Compute neighbors of each top_k1 chunk (over the full corpus)
        primary_embs = embs[top_k1_idx]  # (K1, d)
        # For score propagation: max_k s[k] * cos(k, i) across k in primary
        # = max(primary_embs[k] dot embs[i] * s[k])
        # Vectorize: prop = (primary_embs * s[top_k1_idx, None]) @ embs.T => (K1, N), then max over k.
        weighted = primary_embs * s[top_k1_idx][:, None]  # (K1, d)
        # weighted @ embs.T  (K1, N) — too big for full corpus N=50k
        # do batched max
        prop_score = np.full(embs.shape[0], -np.inf, dtype=np.float32)
        chunk_size = 4096
        for start in range(0, embs.shape[0], chunk_size):
            end = min(start + chunk_size, embs.shape[0])
            block = weighted @ embs[start:end].T  # (K1, block_size)
            prop_score[start:end] = np.maximum(block.max(axis=0), prop_score[start:end])
        # Score amplification: bridge chunks need their propagated score
        # boosted to compete with direct chunks at atom-SNR computation.
        # A bridge chunk has direct_s ~= 0.3, propagated ~= 0.5 (paths via
        # query-similar primary). Amplifying propagation by alpha=2 puts
        # bridge at score ~= 1.0, competitive with direct top (0.7).
        alpha = 2.0
        final = np.maximum(s, alpha * prop_score)
        out_idx = np.argpartition(-final, args.top_n_out)[:args.top_n_out]
        out_idx = out_idx[np.argsort(-final[out_idx])]

        # Build candidate records, preserving structure where possible
        existing_by_id = {c["chunk_id"]: c for c in by_q.get(qid, [])}
        for rank, i in enumerate(out_idx, start=1):
            cid = chunk_ids[int(i)]
            base = existing_by_id.get(cid)
            if base is not None:
                rec = dict(base)
            else:
                rec = {
                    "query_id": qid,
                    "chunk_id": cid,
                    "doc_id": chunk_ids[int(i)].rsplit("::", 1)[0] if "::" in chunk_ids[int(i)] else chunk_ids[int(i)],
                    "retriever": "dense+gwr_2hop",
                    "raw_score": 0.0,
                    "rank": 0,
                    "metadata": {},
                }
            rec["raw_score"] = float(final[i])
            rec["retriever"] = rec.get("retriever", "dense") + "+gwr"
            rec["rank"] = rank
            md = dict(rec.get("metadata", {}))
            md["pre_gwr_score"] = float(s[i])
            md["pre_gwr_rank"] = existing_by_id.get(cid, {}).get("rank")
            rec["metadata"] = md
            new_records.append(rec)
        n_q += 1
        if n_q % 100 == 0:
            print(f"  {n_q}/{len(qids)} done, elapsed {time.time()-t0:.1f}s")

    # If candidate has no doc_id field, look it up
    for rec in new_records:
        if "doc_id" not in rec or not rec["doc_id"]:
            cid = rec["chunk_id"]
            # Look up via chunk_doc map
            pass

    with open(rd / "candidates.jsonl", "w") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0
    with open(rd / "gwr_meta.json", "w") as f:
        json.dump({
            "top_k1": args.top_k1,
            "top_k2": args.top_k2,
            "top_n_out": args.top_n_out,
            "elapsed_seconds": elapsed,
            "n_queries": len(qids),
        }, f, indent=2)
    print(f"GWR done: {len(qids)} queries, {elapsed:.1f}s ({elapsed/len(qids)*1000:.1f}ms/q)")


if __name__ == "__main__":
    main()
