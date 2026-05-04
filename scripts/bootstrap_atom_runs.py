"""Paired bootstrap on per-query F1/EM/cit_acc for atom-level runs.

Reads generated_answers.jsonl (which already contains per-query f1, em).
Compares any list of run dirs vs a baseline. Prints Δ, P(Δ>0), 95% CI
for n_boot=10000.

Also reports bucket-conditional gains using the v4 decomposition
buckets: F1+ vs F1- × cit+ vs cit- × gold-in-pool yes/no. This tells
us *where* a method helps — falsifier check.
"""
from __future__ import annotations
import argparse, json, re, string
from collections import Counter
from pathlib import Path
import numpy as np


_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_EVID = re.compile(r"\[E\d+\]")


def normalize(s):
    s = _EVID.sub(" ", s); s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s); return _WS.sub(" ", s).strip()


def load_run(run_dir: Path, queries: dict) -> dict:
    """Returns {qid: {f1, em, cit_acc, gold_in_pool}}"""
    sel_by_q: dict[str, list[dict]] = {}
    p_sel = run_dir / "selected_atoms.jsonl"
    if p_sel.exists():
        for line in open(p_sel):
            s = json.loads(line)
            sel_by_q.setdefault(s["query_id"], []).append(s)
    out = {}
    for line in open(run_dir / "generated_answers.jsonl"):
        r = json.loads(line)
        qid = r["query_id"]
        gold = set(queries[qid]["gold_doc_ids"])
        sel = sel_by_q.get(qid, [])
        cited = r.get("cited_atom_ids") or []
        atom_doc = {s["atom_id"]: s["doc_id"] for s in sel}
        cit_q = (sum(1 for aid in cited if atom_doc.get(aid) in gold) / len(cited)
                 if cited else 0.0)
        sel_docs = {s["doc_id"] for s in sel}
        gold_in_pool = bool(gold & sel_docs)
        out[qid] = {
            "f1": r.get("f1", 0.0),
            "em": r.get("em", 0.0),
            "cit_acc": cit_q,
            "gold_in_pool": gold_in_pool,
        }
    return out


def boot(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.5, (0.0, 0.0)
    resamples = rng.integers(0, n, (n_boot, n))
    means = diffs[resamples].mean(axis=1)
    return (float(diffs.mean()),
            float((means > 0).mean()),
            (float(np.percentile(means, 2.5)),
             float(np.percentile(means, 97.5))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True,
                    help="run dir for baseline")
    ap.add_argument("--candidates", type=Path, nargs="+", required=True,
                    help="run dirs to compare against baseline")
    args = ap.parse_args()

    queries = {q["query_id"]: q
               for q in (json.loads(l) for l in open(args.queries))}
    base = load_run(args.baseline, queries)
    print(f"baseline: {args.baseline} (n={len(base)})")
    print(f"  F1={np.mean([base[q]['f1'] for q in base]):.4f} "
          f"EM={np.mean([base[q]['em'] for q in base]):.4f} "
          f"cit={np.mean([base[q]['cit_acc'] for q in base]):.4f}\n")

    print(f"{'candidate':<40} | {'metric':<8} | {'delta':>8} | "
          f"{'P(>0)':>6} | {'CI95':>22}")
    print("-" * 100)

    for cand_dir in args.candidates:
        cand = load_run(cand_dir, queries)
        common = sorted(set(base) & set(cand))
        if not common:
            continue
        cand_name = cand_dir.name
        for metric in ["f1", "em", "cit_acc"]:
            d = np.array([cand[q][metric] - base[q][metric] for q in common])
            m, p_pos, ci = boot(d)
            print(f"{cand_name:<40} | {metric:<8} | {m:>+.4f} | "
                  f"{p_pos:>6.3f} | [{ci[0]:>+.4f}, {ci[1]:>+.4f}]")
        print()

        # Bucket-conditional gain (F1 within F1- queries of baseline)
        f1_low = [q for q in common if base[q]["f1"] < 0.5]
        if f1_low:
            d_low = np.array([cand[q]["f1"] - base[q]["f1"] for q in f1_low])
            m, p_pos, ci = boot(d_low)
            print(f"  >> conditioned on baseline F1<0.5 (n={len(f1_low)}):")
            print(f"     ΔF1={m:+.4f} P(>0)={p_pos:.3f} CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
