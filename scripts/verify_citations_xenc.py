"""Cross-encoder-based citation verification (NLI-style baseline).

Comparison to the +11.7pp trivial token-overlap fix. Uses
bge-reranker-v2-m3 (already cached) as proxy NLI: for each cited atom,
score (atom_text, answer_text). Drop if score < threshold.

Output: cit_acc as a function of threshold; finds whether sophisticated
NLI-style scoring beats the trivial overlap filter.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    queries = {q["query_id"]: q
               for q in (json.loads(l) for l in open(args.queries))}
    sel_by_q: dict[str, dict[str, dict]] = {}
    for line in open(args.run_dir / "selected_atoms.jsonl"):
        s = json.loads(line)
        sel_by_q.setdefault(s["query_id"], {})[s["atom_id"]] = s
    answers = [json.loads(l)
               for l in open(args.run_dir / "generated_answers.jsonl")]

    pairs: list[tuple[str, str]] = []
    pair_meta: list[tuple[int, str, str]] = []  # (answer_idx, qid, atom_id)
    for ai, r in enumerate(answers):
        cited = r.get("cited_atom_ids") or []
        ans_text = r.get("answer_text") or ""
        if not ans_text or not cited:
            continue
        atom_by_id = sel_by_q.get(r["query_id"], {})
        for aid in cited:
            atom = atom_by_id.get(aid)
            if atom is None:
                continue
            pairs.append((atom["text"], ans_text))
            pair_meta.append((ai, r["query_id"], aid))
    print(f"scoring {len(pairs)} (atom, answer) pairs")

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(args.reranker_model, device=args.device)
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    scores = np.asarray(scores, dtype=np.float32)

    # cit_acc as function of threshold
    thresholds = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
    rows = []
    cit_per_pair = []
    for k, thr in enumerate(thresholds):
        kept = scores >= thr
        # group by query
        cit_sum = 0.0; n_q = 0
        by_q: dict[str, list[tuple[int, float]]] = {}
        for (ai, qid, aid), keep, sc in zip(pair_meta, kept, scores):
            by_q.setdefault(qid, []).append((1 if keep else 0, sc))
        for qid, items in by_q.items():
            atoms_kept = [(aid, sc) for (ai_x, qid_x, aid), keep, sc
                          in zip(pair_meta, kept, scores)
                          if qid_x == qid and keep]
            if not atoms_kept:
                continue
            gold = set(queries[qid]["gold_doc_ids"])
            atom_doc = {aid: sel_by_q[qid][aid]["doc_id"] for aid, _ in atoms_kept
                        if aid in sel_by_q[qid]}
            if atom_doc:
                hit = sum(1 for d in atom_doc.values() if d in gold)
                cit_sum += hit / len(atom_doc)
                n_q += 1
        cit_acc = cit_sum / max(1, n_q)
        rows.append((thr, cit_acc, int(kept.sum()), n_q))
        print(f"thr={thr:5.2f}  cit_acc={cit_acc:.4f}  "
              f"kept={int(kept.sum())}/{len(pairs)}  n_q_with_cite={n_q}")

    # Save
    with open(args.run_dir / "metrics_xenc_verify.json", "w") as f:
        json.dump({
            "n_pairs": len(pairs),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "thresholds": rows,
        }, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
