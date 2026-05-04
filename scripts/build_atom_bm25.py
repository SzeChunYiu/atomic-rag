"""Build BM25 index over atoms (sentences) for hybrid sparse+dense.

Existing BM25 covers chunks only. Atom-level BM25 enables:
- Sparse-only baseline (lexical match on sentences, no embedder)
- Hybrid fusion (dense + BM25) with linear/RRF combination
- Different signal than dense; helps queries with rare entities
  poorly captured by BGE-M3

Output: atom_bm25.npz with sparse term-document matrix in CSR format,
plus vocab.json mapping term -> column index.
"""
from __future__ import annotations
import argparse, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse


_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b")
# minimal english stoplist (BM25 typically uses one)
_STOP = set(("a an the of in on at to for from with by is are was were be been "
             "being and or but if then so than that this these those it its as "
             "i you he she we they me him her us them my your his their our "
             "what when where why how which who whom whose").split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text) if t.lower() not in _STOP]


def build_bm25(
    docs: list[list[str]], k1: float = 1.2, b: float = 0.75
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    """Returns (tfidf_csr, vocab, idf_array)."""
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))
    n_docs = len(docs)
    vocab = {t: i for i, (t, _) in enumerate(
        sorted(df.items(), key=lambda x: -x[1]))}
    n_terms = len(vocab)
    idf = np.zeros(n_terms, dtype=np.float32)
    for t, i in vocab.items():
        idf[i] = math.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    rows = []; cols = []; data = []
    for d_i, tokens in enumerate(docs):
        if not tokens: continue
        tf = Counter(tokens)
        dl = len(tokens)
        for term, freq in tf.items():
            j = vocab.get(term)
            if j is None: continue
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score = idf[j] * (freq * (k1 + 1)) / denom
            rows.append(d_i); cols.append(j); data.append(score)
    csr = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32)
    return csr, vocab, idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    print(f"loaded {len(atoms)} atoms")
    docs = [tokenize(a["text"]) for a in atoms]
    csr, vocab, idf = build_bm25(docs)
    print(f"BM25: {len(vocab)} unique terms; matrix {csr.shape}, "
          f"nnz={csr.nnz}")
    sparse.save_npz(args.out_dir / "atom_bm25.npz", csr)
    np.save(args.out_dir / "atom_bm25_idf.npy", idf)
    with open(args.out_dir / "atom_bm25_vocab.json", "w") as f:
        json.dump(vocab, f)
    print(f"wrote to {args.out_dir}")


if __name__ == "__main__":
    main()
