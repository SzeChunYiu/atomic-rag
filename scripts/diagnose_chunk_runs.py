"""Per-query failure-mode diagnostic for chunk-level runs.

Decomposes failures into:
  (F1 high/low) x (cit_acc high/low) x (gold_doc_in_pool) x (gold_text_in_pool)

Failure taxonomy:
  F1-  cit-  gold-  text-  → retrieval miss (gold not retrieved)
  F1-  cit-  gold+  text-  → selection miss (gold retrieved but not selected)
  F1-  cit-  gold+  text+  → generation miss (gold text in context but answer wrong)
  F1-  cit+  gold+  text+  → citation ok but generation wrong
  F1+  *     *      *      → success (may have various citation quality)

Usage:
    python scripts/diagnose_chunk_runs.py --run-dir RUNDIR --queries QUERIES \
        --chunks CHUNKS [--dataset DATASET]
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

_CITE_RE = re.compile(r"\[E\d+\]")
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_STOP = set(
    "a an the of in on at to for from with by is are was were be been "
    "being and or but if then so than that this these those it its as "
    "which who whom whose what when where why how".split()
)
_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_clean(s: str) -> str:
    return norm(_CITE_RE.sub("", s))


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def token_f1(pred: str, refs: list[str]) -> float:
    pt = norm_clean(pred).split()
    if not pt:
        return 0.0
    best = 0.0
    for r in refs:
        rt = norm(r).split()
        if not rt:
            continue
        common = Counter(pt) & Counter(rt)
        nc = sum(common.values())
        if nc == 0:
            continue
        p = nc / len(pt)
        rec = nc / len(rt)
        best = max(best, 2 * p * rec / (p + rec))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument("--cit-threshold", type=float, default=0.5)
    args = ap.parse_args()

    # Load queries
    queries: dict[str, dict] = {}
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        queries[r["query_id"]] = r

    # Load chunk → doc mapping and chunk text
    chunk_doc: dict[str, str] = {}
    chunk_text: dict[str, str] = {}
    for line in args.chunks.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cid = r["chunk_id"]
        chunk_doc[cid] = r["doc_id"]
        chunk_text[cid] = r.get("text", "")

    # Load selected context per query
    sel_by_q: dict[str, list[str]] = {}
    sel_path = args.run_dir / "selected_context.jsonl"
    if sel_path.is_file():
        for line in sel_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sel_by_q.setdefault(r["query_id"], []).append(r["chunk_id"])

    # Load answers
    ans_path = args.run_dir / "generated_answers.jsonl"
    answers = []
    for line in ans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        answers.append(json.loads(line))

    bucket: Counter = Counter()
    f1_low_gold_in_pool = 0
    f1_low_gold_text_in_pool = 0
    bridge_in_pool_n = 0
    bridge_total = 0
    f1_cit_pairs = []

    for row in answers:
        qid = row["query_id"]
        q = queries.get(qid)
        if q is None:
            continue

        gold_doc_ids = set(q.get("gold_doc_ids") or [])
        gold_ans = q.get("metadata", {}).get("answer") or ""
        if isinstance(gold_ans, list):
            gold_ans = " ".join(gold_ans)
        refs = [gold_ans] if gold_ans else []

        pred = row.get("answer_text", "")
        f1 = token_f1(pred, refs) if refs else 0.0

        selected = sel_by_q.get(qid) or row.get("selected_chunk_ids") or []
        cited = row.get("cited_chunk_ids") or []

        selected_docs = {chunk_doc[c] for c in selected if c in chunk_doc}

        # Bridge coverage: both gold docs in selected pool
        if len(gold_doc_ids) >= 2:
            bridge_total += 1
            if gold_doc_ids.issubset(selected_docs):
                bridge_in_pool_n += 1

        gold_in_pool = bool(gold_doc_ids & selected_docs)

        # Gold answer text present in any selected chunk
        gold_tok = content_tokens(gold_ans)
        gold_text_in_pool = False
        if gold_tok:
            for cid in selected:
                if gold_tok.issubset(content_tokens(chunk_text.get(cid, ""))):
                    gold_text_in_pool = True
                    break

        # Citation accuracy
        cit_q = 0.0
        if cited and gold_doc_ids:
            hits = sum(1 for c in cited if chunk_doc.get(c) in gold_doc_ids)
            cit_q = hits / len(cited)

        f1_hi = f1 >= args.f1_threshold
        cit_hi = cit_q >= args.cit_threshold
        bucket[(f1_hi, cit_hi, gold_in_pool, gold_text_in_pool)] += 1
        f1_cit_pairs.append((f1, cit_q))

        if not f1_hi and gold_in_pool:
            f1_low_gold_in_pool += 1
            if gold_text_in_pool:
                f1_low_gold_text_in_pool += 1

    total = sum(bucket.values())
    n_success = sum(v for (f1_hi, *_), v in bucket.items() if f1_hi)

    print(f"\n=== Failure Decomposition ({args.run_dir.name}) ===")
    print(f"Total queries: {total}")
    print(f"F1 >= {args.f1_threshold}: {n_success} ({n_success/total:.1%})")
    print(f"F1 <  {args.f1_threshold}: {total - n_success} ({(total-n_success)/total:.1%})")

    if bridge_total > 0:
        print(f"\nBridge coverage (both gold docs in selected pool): "
              f"{bridge_in_pool_n}/{bridge_total} ({bridge_in_pool_n/bridge_total:.1%})")

    print(f"\nBucket breakdown (F1≥{args.f1_threshold}, cit≥{args.cit_threshold}, "
          f"gold_doc_in_pool, gold_text_in_pool):")
    for k, v in sorted(bucket.items(), key=lambda x: -x[1]):
        f1_hi, cit_hi, gp, tp = k
        label = (
            f"F1{'✓' if f1_hi else '✗'} "
            f"cit{'✓' if cit_hi else '✗'} "
            f"gold_doc{'✓' if gp else '✗'} "
            f"gold_text{'✓' if tp else '✗'}"
        )
        failure_mode = ""
        if not f1_hi:
            if not gp:
                failure_mode = " ← RETRIEVAL MISS"
            elif not tp:
                failure_mode = " ← SELECTION MISS"
            else:
                failure_mode = " ← GENERATION MISS"
        print(f"  {label}: {v:4d} ({v/total:.1%}){failure_mode}")

    print(f"\nF1<{args.f1_threshold} with gold doc in pool: {f1_low_gold_in_pool} "
          f"({f1_low_gold_in_pool/total:.1%})")
    print(f"  ... AND gold answer text in pool: {f1_low_gold_text_in_pool} "
          f"({f1_low_gold_text_in_pool/total:.1%})")

    # Mean F1 conditional on gold presence
    f1_gold_in = [f1 for (f1, _), (qid, q) in
                  [(fp, (row["query_id"], queries.get(row["query_id"])))
                   for fp, row in zip(f1_cit_pairs, answers)]
                  if q and (set(q.get("gold_doc_ids") or [])) &
                  {chunk_doc[c] for c in
                   (sel_by_q.get(qid) or row.get("selected_chunk_ids") or [])
                   if c in chunk_doc}]
    if f1_gold_in:
        print(f"\nMean F1 when gold doc in pool: {sum(f1_gold_in)/len(f1_gold_in):.4f} "
              f"(n={len(f1_gold_in)})")

    if f1_cit_pairs:
        try:
            from statistics import correlation
            r = correlation([p[0] for p in f1_cit_pairs], [p[1] for p in f1_cit_pairs])
            print(f"F1 vs cit_acc Pearson r: {r:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
