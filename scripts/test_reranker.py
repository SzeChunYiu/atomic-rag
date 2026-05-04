"""Stage diagnostic: reranker only.

Takes a pre-retrieved candidate pool (candidates.jsonl) and re-scores with
BGE-reranker-v2-m3. Reports: MRR, Recall@K before vs after reranking, and
cases where reranking *hurts* (gold chunk rank goes down).

Usage:
  python scripts/test_reranker.py \
    --candidates candidates.jsonl \
    --queries queries.jsonl \
    --n 100
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    return float(bool(gold_ids & set(ranked_ids[:k])))


def mrr(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for i, cid in enumerate(ranked_ids):
        if cid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True,
                    help="candidates.jsonl — each line: {query_id, chunk_id, doc_id, score, text}")
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries))}

    cands_by_q: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.candidates):
        c = json.loads(line)
        cands_by_q[c["query_id"]].append(c)

    qids = list(cands_by_q)[:args.n]
    print(f"Queries: {len(qids)}  reranker: {args.model}  top_k: {args.top_k}")

    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(args.model, use_fp16=True)
    print("Reranker loaded.\n")

    before_r1, after_r1, before_mrr, after_mrr = [], [], [], []
    hurt_count = 0

    for qid in qids:
        q = queries.get(qid)
        if q is None:
            continue
        gold = set(q.get("gold_doc_ids", []))
        cands = cands_by_q[qid]

        # Chunks whose doc_id is gold
        gold_chunk_ids = {c["chunk_id"] for c in cands if c.get("doc_id") in gold}

        before_ranked = [c["chunk_id"] for c in
                         sorted(cands, key=lambda x: -x.get("score", 0))]

        pairs = [[q.get("query") or q.get("text", ""), c["text"]] for c in cands]
        scores = reranker.compute_score(pairs, normalize=True)

        after_ranked = [c["chunk_id"] for c, _ in
                        sorted(zip(cands, scores), key=lambda x: -x[1])]

        b_r1 = recall_at_k(before_ranked, gold_chunk_ids, args.top_k)
        a_r1 = recall_at_k(after_ranked, gold_chunk_ids, args.top_k)
        b_mrr = mrr(before_ranked, gold_chunk_ids)
        a_mrr = mrr(after_ranked, gold_chunk_ids)

        before_r1.append(b_r1); after_r1.append(a_r1)
        before_mrr.append(b_mrr); after_mrr.append(a_mrr)
        if a_mrr < b_mrr:
            hurt_count += 1

    n = len(before_r1)
    print(f"=== Reranker diagnostic ({n} queries) ===")
    print(f"  Recall@{args.top_k} before: {sum(before_r1)/n:.3f}  after: {sum(after_r1)/n:.3f}  "
          f"delta: {(sum(after_r1)-sum(before_r1))/n:+.3f}")
    print(f"  MRR      before: {sum(before_mrr)/n:.3f}  after: {sum(after_mrr)/n:.3f}  "
          f"delta: {(sum(after_mrr)-sum(before_mrr))/n:+.3f}")
    print(f"  Reranker hurts (MRR drops): {hurt_count} ({hurt_count/n:.1%})")


if __name__ == "__main__":
    main()
