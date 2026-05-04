"""Build a typed-atom catalog from an existing index_bundle.

Reads:
  index_bundle/chunks.jsonl
  index_bundle/embeddings.npy
  (uses same embedder via index_meta.json)

Writes:
  atoms_dir/atoms.jsonl       # atom_id, chunk_id, doc_id, text, claim_type, span_start, span_end
  atoms_dir/atom_embs.npy     # (N_atoms, d)
  atoms_dir/atom_meta.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")
sys.path.insert(0, "/Users/billy/Desktop/projects/AI_engineering/RAG/src")
from astro_cs_rag.atoms.deblend import deblend_chunk, type_tag, query_intent
from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
from astro_cs_rag.config.schema import EmbeddingSettings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir", type=Path)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    idx = args.index_dir
    out_dir = args.out_dir or (idx / "atoms")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks = []
    for line in open(idx / "chunks.jsonl"):
        c = json.loads(line)
        chunks.append(c)
    print(f"Loaded {len(chunks)} chunks")

    # Deblend all chunks
    t0 = time.time()
    atoms = []
    for c in chunks:
        atoms.extend(deblend_chunk(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            chunk_text=c["text"],
        ))
    print(f"Deblended into {len(atoms)} atoms in {time.time()-t0:.1f}s")

    # Embed atoms in batches
    _, _, _, meta = load_index_bundle(idx)
    embedder = embedder_from_meta(meta, EmbeddingSettings())

    t1 = time.time()
    texts = [a.text for a in atoms]
    all_embs = []
    for start in range(0, len(texts), args.batch_size):
        end = min(start + args.batch_size, len(texts))
        embs = embedder.encode(texts[start:end])
        all_embs.append(embs.astype(np.float32))
        if (start // args.batch_size) % 50 == 0:
            print(f"  embedded {end}/{len(texts)} elapsed={time.time()-t1:.1f}s")
    atom_embs = np.vstack(all_embs)
    norms = np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-9
    atom_embs = (atom_embs / norms).astype(np.float32)
    print(f"Embedded {len(atoms)} atoms in {time.time()-t1:.1f}s, shape={atom_embs.shape}")

    # Write
    np.save(out_dir / "atom_embs.npy", atom_embs)
    with open(out_dir / "atoms.jsonl", "w") as f:
        for a in atoms:
            f.write(json.dumps({
                "atom_id": a.atom_id,
                "chunk_id": a.chunk_id,
                "doc_id": a.doc_id,
                "text": a.text,
                "claim_type": a.claim_type,
                "claim_type_conf": a.claim_type_conf,
                "span_start": a.span_start,
                "span_end": a.span_end,
            }) + "\n")
    with open(out_dir / "atom_meta.json", "w") as f:
        json.dump({
            "n_atoms": len(atoms),
            "n_chunks": len(chunks),
            "atoms_per_chunk_mean": len(atoms) / max(1, len(chunks)),
            "claim_type_counts": {t: sum(1 for a in atoms if a.claim_type == t)
                                  for t in ["WHO", "WHEN", "WHERE", "WHAT_NUM",
                                            "WHAT_OBJ", "ANY"]},
        }, f, indent=2)
    print(f"Wrote {out_dir}/atoms.jsonl ({len(atoms)}) + atom_embs.npy")


if __name__ == "__main__":
    main()
