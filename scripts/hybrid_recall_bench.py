"""Hybrid BM25+dense atom retrieval — recall-only benchmark (CPU).

Tests whether sparse+dense fusion improves bridge-doc recall (B2).
Doesn't need generation; just measures candidate-pool recall against
gold docs at multiple K's.

Fusion methods:
- linear: alpha * dense + (1-alpha) * bm25_norm
- rrf: reciprocal-rank fusion (Cormack-Clarke-Buettcher 2009)

Output: per-method recall@10/30/50/100 and bridge_recall (both gold
docs in pool) at each K.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import sparse


_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b")
_STOP = set(("a an the of in on at to for from with by is are was were be been "
             "being and or but if then so than that this these those it its as "
             "i you he she we they me him her us them my your his their our "
             "what when where why how which who whom whose").split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text) if t.lower() not in _STOP]


def bm25_query_score(
    query_tokens: list[str], bm25_csr: sparse.csr_matrix,
    vocab: dict[str, int], idf: np.ndarray,
) -> np.ndarray:
    """Return (n_atoms,) BM25 score = sum over query terms of doc-term weight."""
    cols = [vocab[t] for t in query_tokens if t in vocab]
    if not cols:
        return np.zeros(bm25_csr.shape[0], dtype=np.float32)
    return np.asarray(bm25_csr[:, cols].sum(axis=1)).flatten()


def rrf_combine(
    ranks_lists: list[np.ndarray], k: int = 60
) -> np.ndarray:
    """Reciprocal-rank fusion. Each ranks_lists[i] = atom ranks (lower=better)."""
    n = ranks_lists[0].shape[0]
    rrf = np.zeros(n, dtype=np.float32)
    for ranks in ranks_lists:
        rrf += 1.0 / (k + ranks)
    return rrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    atom_doc = np.array([a["doc_id"] for a in atoms])
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    atom_embs = atom_embs / (np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-9)

    bm25 = sparse.load_npz(args.atoms_dir / "atom_bm25.npz")
    idf = np.load(args.atoms_dir / "atom_bm25_idf.npy")
    vocab = json.loads((args.atoms_dir / "atom_bm25_vocab.json").read_text())
    print(f"loaded {len(atoms)} atoms, BM25 vocab size {len(vocab)}")

    # Need query embeddings — do CPU encode via sentence-transformers
    # or load existing if available. For CPU-only we'll skip dense and
    # use BM25-only here, but re-load existing q_embs if cached.
    q_embs_path = args.atoms_dir / ".cached_q_embs.npy"
    if q_embs_path.exists():
        q_embs = np.load(q_embs_path)
        queries = [json.loads(l) for l in open(args.queries)]
        print(f"loaded {len(q_embs)} cached query embeddings")
    else:
        # Encode on CPU — slow but works for 1000 queries
        from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
        from astro_cs_rag.config.schema import EmbeddingSettings
        _, _, _, meta = load_index_bundle(args.index_dir)
        embedder = embedder_from_meta(meta, EmbeddingSettings())
        queries = [json.loads(l) for l in open(args.queries)]
        print(f"encoding {len(queries)} queries on CPU...")
        q_embs = embedder.encode([q["text"] for q in queries]).astype(np.float32)
        q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
        np.save(q_embs_path, q_embs)

    # Per-query: dense scores, BM25 scores, fusion scores
    K_list = [10, 30, 50, 100]
    methods = ["dense", "bm25", "linear_a0.5", "linear_a0.7", "rrf"]
    recall_per_q: dict[str, dict[int, list[float]]] = {
        m: {K: [] for K in K_list} for m in methods}
    bridge_per_q: dict[str, dict[int, list[float]]] = {
        m: {K: [] for K in K_list} for m in methods}

    for qi, q in enumerate(queries):
        gold = set(q["gold_doc_ids"])
        # Dense
        d_scores = atom_embs @ q_embs[qi]
        # BM25
        b_scores = bm25_query_score(tokenize(q["text"]), bm25, vocab, idf)
        # Linear fusion (normalize each)
        d_n = (d_scores - d_scores.mean()) / (d_scores.std() + 1e-9)
        b_n = (b_scores - b_scores.mean()) / (b_scores.std() + 1e-9)
        l05 = 0.5 * d_n + 0.5 * b_n
        l07 = 0.7 * d_n + 0.3 * b_n
        # RRF
        d_ranks = np.argsort(-d_scores).argsort()
        b_ranks = np.argsort(-b_scores).argsort()
        rrf = rrf_combine([d_ranks, b_ranks])

        for name, sc in [("dense", d_scores), ("bm25", b_scores),
                         ("linear_a0.5", l05), ("linear_a0.7", l07),
                         ("rrf", rrf)]:
            order = np.argsort(-sc)
            for K in K_list:
                top = order[:K]
                docs_in_pool = set(atom_doc[top])
                hit = len(gold & docs_in_pool)
                recall_per_q[name][K].append(hit / max(1, len(gold)))
                bridge_per_q[name][K].append(
                    1.0 if gold.issubset(docs_in_pool) else 0.0)

    # Aggregate
    rows = []
    print(f"\n{'method':<14} | " + " | ".join(f"R@{K}" for K in K_list)
          + " | " + " | ".join(f"Br@{K}" for K in K_list))
    print("-" * 110)
    for m in methods:
        rec_mean = [np.mean(recall_per_q[m][K]) for K in K_list]
        bri_mean = [np.mean(bridge_per_q[m][K]) for K in K_list]
        print(f"{m:<14} | " + " | ".join(f"{r:.3f}" for r in rec_mean)
              + " | " + " | ".join(f"{b:.3f}" for b in bri_mean))
        rows.append({"method": m,
                     **{f"R@{K}": r for K, r in zip(K_list, rec_mean)},
                     **{f"Br@{K}": b for K, b in zip(K_list, bri_mean)}})

    with open(args.out_dir / "hybrid_recall.json", "w") as f:
        json.dump({"K_list": K_list, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out_dir/'hybrid_recall.json'}")


if __name__ == "__main__":
    main()
