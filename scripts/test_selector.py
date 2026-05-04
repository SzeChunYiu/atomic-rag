"""Stage diagnostic: selector only.

Takes a candidate pool and runs each selector (greedy, mmr, anti_kt, maxent,
submodular) with a given token budget. Reports: gold-chunk recall per selector,
budget utilization, and per-query winner.

Usage:
  python scripts/test_selector.py \
    --candidates candidates.jsonl \
    --queries queries.jsonl \
    --budget 1024 --n 100
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def recall(selected: list[dict], gold_chunk_ids: set[str]) -> float:
    sel_ids = {s.get("chunk_id") or s.get("atom_id", "") for s in selected}
    return float(bool(gold_chunk_ids & sel_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--selectors", nargs="*",
                    default=["greedy", "mmr", "anti_kt", "submodular", "maxent"])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from astro_cs_rag.selection.submodular import submodular_select, query_facets
    from astro_cs_rag.selection.maxent import maxent_select
    from astro_cs_rag.selection.anti_kt import select_evidence_via_jets

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries))}

    cands_by_q: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.candidates):
        c = json.loads(line)
        # Normalize field name for selector compatibility
        if "chunk_id" in c and "atom_id" not in c:
            c["atom_id"] = c["chunk_id"]
        if "claim_type" not in c:
            c["claim_type"] = "ANY"
        cands_by_q[c["query_id"]].append(c)

    qids = list(cands_by_q)[:args.n]
    print(f"Queries: {len(qids)}  budget: {args.budget}  selectors: {args.selectors}\n")

    recall_by_sel: dict[str, list[float]] = {s: [] for s in args.selectors}
    budget_used_by_sel: dict[str, list[int]] = {s: [] for s in args.selectors}

    for qid in qids:
        q = queries.get(qid)
        if q is None:
            continue
        gold_docs = set(q.get("gold_doc_ids", []))
        cands = cands_by_q[qid]
        gold_chunks = {c.get("chunk_id", c.get("atom_id", ""))
                       for c in cands if c.get("doc_id") in gold_docs}

        # Sort candidates by score descending (greedy baseline)
        ranked = sorted(cands, key=lambda x: -x.get("score", 0))
        facets = query_facets(q.get("query") or q.get("text", ""), "ANY")

        for sel_name in args.selectors:
            if sel_name == "greedy":
                selected, used = [], 0
                for c in ranked:
                    t = c.get("token_count") or max(1, len(c["text"].split()))
                    if used + t <= args.budget:
                        selected.append(c); used += t
            elif sel_name == "mmr":
                # Simple MMR: score - lambda * max_sim_to_selected
                lam = 0.7
                selected, used, sel_texts = [], 0, []
                remaining = list(ranked)
                while remaining:
                    best, best_score, best_i = None, -1e9, -1
                    for i, c in enumerate(remaining):
                        t = c.get("token_count") or max(1, len(c["text"].split()))
                        if used + t > args.budget:
                            continue
                        sim_to_sel = max(
                            (len(set(c["text"].lower().split()) &
                                 set(s.lower().split())) /
                             max(1, len(set(c["text"].lower().split())))
                             for s in sel_texts), default=0.0)
                        score = lam * c.get("score", 0) - (1 - lam) * sim_to_sel
                        if score > best_score:
                            best_score, best, best_i = score, c, i
                    if best is None:
                        break
                    selected.append(best)
                    used += best.get("token_count") or max(1, len(best["text"].split()))
                    sel_texts.append(best["text"])
                    remaining.pop(best_i)
            elif sel_name == "submodular":
                selected = submodular_select(
                    atoms=cands, facets=facets, token_budget=args.budget)
                used = sum(c.get("token_count") or max(1, len(c["text"].split()))
                           for c in selected)
            elif sel_name == "maxent":
                selected = maxent_select(
                    atoms=cands, facets=facets, token_budget=args.budget)
                used = sum(c.get("token_count") or max(1, len(c["text"].split()))
                           for c in selected)
            elif sel_name == "anti_kt":
                import numpy as np
                ids = [c.get("atom_id", c.get("chunk_id", str(i)))
                       for i, c in enumerate(cands)]
                rels = [max(1e-3, c.get("score", 0.1)) for c in cands]
                # Use random embeddings if real ones absent (smoke test only)
                if cands[0].get("embedding"):
                    embs = np.array([c["embedding"] for c in cands], dtype=np.float32)
                else:
                    rng = __import__("numpy").random.default_rng(0)
                    embs = rng.standard_normal((len(cands), 64)).astype("float32")
                    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
                kept_ids = set(select_evidence_via_jets(ids, rels, embs, n_jets=3))
                selected = [c for c in ranked
                            if c.get("atom_id", c.get("chunk_id", "")) in kept_ids]
                used = sum(c.get("token_count") or max(1, len(c["text"].split()))
                           for c in selected)
            else:
                selected, used = [], 0

            rec = recall(selected, gold_chunks)
            recall_by_sel[sel_name].append(rec)
            budget_used_by_sel[sel_name].append(used)

    print(f"{'Selector':<12}  {'Recall@budget':>14}  {'Avg tokens used':>16}")
    print("-" * 48)
    for sel_name in args.selectors:
        rs = recall_by_sel[sel_name]
        bu = budget_used_by_sel[sel_name]
        n = len(rs)
        print(f"{sel_name:<12}  {sum(rs)/n:>14.3f}  {sum(bu)/n:>16.0f}")


if __name__ == "__main__":
    main()
