"""Per-query failure-mode diagnostic.

For D04+D06 CoT, computes joint distribution:
  (F1 high/low) x (cit_acc high/low) x (gold-doc-in-pool yes/no)

This tells us where the F1 loss actually comes from:
- F1 high, cit low, gold in pool -> generator right despite hallucinated cites
- F1 low, cit high, gold in pool -> generation bottleneck (gold present but unused)
- F1 low, cit low, gold NOT in pool -> retrieval failure (B2)
- F1 low, cit low, gold IN pool, span correct -> selection failure
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter


_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")
_STOP = set("a an the of in on at to for from with by is are was were be been "
            "being and or but if then so than that this these those it its as "
            "which who whom whose what when where why how".split())


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries))}
    # Build chunk→doc map from selected_context.jsonl (chunk-level pipeline).
    # Fall back to selected_atoms.jsonl for the legacy atom pipeline.
    sel_by_q: dict[str, list[dict]] = {}
    _sel_path = args.run_dir / "selected_context.jsonl"
    if not _sel_path.is_file():
        _sel_path = args.run_dir / "selected_atoms.jsonl"
    _chunks_path = args.run_dir.parent / "index_bundle" / "chunks.jsonl"
    _cid_to_doc: dict[str, str] = {}
    if _chunks_path.is_file():
        for _line in open(_chunks_path):
            _c = json.loads(_line)
            _cid_to_doc[_c["chunk_id"]] = _c["doc_id"]
    for line in open(_sel_path):
        s = json.loads(line)
        cid = s.get("chunk_id", "")
        doc_id = s.get("doc_id") or _cid_to_doc.get(cid, "")
        sel_by_q.setdefault(s["query_id"], []).append({**s, "doc_id": doc_id})
    answers = [json.loads(l)
               for l in open(args.run_dir / "generated_answers.jsonl")]

    bucket = Counter()
    f1_low_with_gold = 0; f1_low_with_gold_text_present = 0
    cit_independence = []
    bridge_in_pool_count = 0; bridge_total = 0

    for r in answers:
        qid = r["query_id"]
        q = queries[qid]
        gold = set(q["gold_doc_ids"])
        sel = sel_by_q.get(qid, [])
        sel_docs = {s["doc_id"] for s in sel}
        ans = r.get("answer_text") or ""
        ans_tok = content_tokens(ans)
        f1 = r["f1"]

        # bridge_in_pool: both gold docs present in selected pool
        if len(gold) >= 2:
            bridge_total += 1
            if gold.issubset(sel_docs):
                bridge_in_pool_count += 1

        # gold in selection at all
        gold_in_pool = bool(gold & sel_docs)

        # cit_acc per query
        cited = r.get("cited_atom_ids") or []
        cit_q = 0.0
        if cited:
            atom_doc = {s["atom_id"]: s["doc_id"] for s in sel}
            n_hit = sum(1 for aid in cited if atom_doc.get(aid) in gold)
            cit_q = n_hit / len(cited)

        # gold answer text present in selection (atom-level)
        gold_ans = q["metadata"].get("answer") or ""
        if isinstance(gold_ans, list): gold_ans = " ".join(gold_ans)
        gold_tok = content_tokens(gold_ans)
        gold_text_in_pool = False
        if gold_tok and sel:
            for s in sel:
                if gold_tok.issubset(content_tokens(s["text"])):
                    gold_text_in_pool = True
                    break

        # bucket assignment
        f1_hi = f1 >= 0.5
        cit_hi = cit_q >= 0.5
        b = (f1_hi, cit_hi, gold_in_pool, gold_text_in_pool)
        bucket[b] += 1
        cit_independence.append((f1, cit_q))

        if not f1_hi and gold_in_pool:
            f1_low_with_gold += 1
            if gold_text_in_pool:
                f1_low_with_gold_text_present += 1

    print(f"\nbridge_in_pool: {bridge_in_pool_count}/{bridge_total} "
          f"({bridge_in_pool_count/max(1,bridge_total):.1%})")

    print("\nQuery buckets (F1>=0.5, cit>=0.5, gold_doc_in_pool, "
          "gold_answer_text_in_pool):")
    rows = sorted(bucket.items(), key=lambda x: -x[1])
    total = sum(bucket.values())
    for k, v in rows:
        f1_hi, cit_hi, gold_p, text_p = k
        print(f"  F1{'+' if f1_hi else '-'} "
              f"cit{'+' if cit_hi else '-'} "
              f"gold{'+' if gold_p else '-'} "
              f"text{'+' if text_p else '-'} : "
              f"{v} ({v/total:.1%})")

    print(f"\nF1<0.5 with gold doc in pool: {f1_low_with_gold} "
          f"({f1_low_with_gold/total:.1%})")
    print(f"  ... AND gold answer text in pool: "
          f"{f1_low_with_gold_text_present} "
          f"({f1_low_with_gold_text_present/total:.1%})")

    # Correlation F1 vs cit_acc
    pairs = [(f1, c) for f1, c in cit_independence if c > 0 or f1 > 0]
    if pairs:
        from statistics import correlation
        try:
            r = correlation([p[0] for p in pairs], [p[1] for p in pairs])
            print(f"\nF1 vs cit_acc correlation: {r:.3f} (n={len(pairs)})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
