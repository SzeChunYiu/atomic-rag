"""Citation Post-Verification (CPV).

Post-hoc fix for cited_chunk_ids in a run directory. After the generator
emits (answer_text, cited_chunk_ids), we verify each citation actually
contains content words from the answer; if not, we replace the citation
with a chunk in selected_context that does.

Usage:
    python cpv_apply.py <run_dir> [--out generated_answers_cpv.jsonl]

Recomputes citation_accuracy on the fixed citations.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter

PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")
EVID_RE = re.compile(r"\[E\d+\]")
STOP = {"a", "an", "the", "is", "was", "are", "were", "of", "in", "on", "at", "to",
        "for", "and", "or", "by", "with", "from", "as", "i", "dont", "know"}


def content_tokens(answer: str) -> set[str]:
    s = EVID_RE.sub(" ", answer)
    s = PUNCT_RE.sub(" ", s.lower())
    return {tok for tok in s.split() if tok and tok not in STOP and len(tok) > 1}


def chunk_supports(chunk_text: str, ans_tokens: set[str]) -> int:
    """Return # of answer content tokens present in chunk text."""
    if not ans_tokens:
        return 0
    s = PUNCT_RE.sub(" ", chunk_text.lower())
    bag = set(s.split())
    return len(ans_tokens & bag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--out_name", default="generated_answers_cpv.jsonl")
    args = ap.parse_args()

    run_dir = args.run_dir
    parent = run_dir.parent  # has index_bundle/

    chunks_path = parent / "index_bundle" / "chunks.jsonl"
    chunk_text: dict[str, str] = {}
    chunk_doc: dict[str, str] = {}
    for line in open(chunks_path):
        c = json.loads(line)
        chunk_text[c["chunk_id"]] = c["text"]
        chunk_doc[c["chunk_id"]] = c["doc_id"]

    sel: dict[str, list[str]] = {}
    for line in open(run_dir / "selected_context.jsonl"):
        r = json.loads(line)
        sel.setdefault(r["query_id"], []).append(r["chunk_id"])

    out_rows = []
    n_replaced = 0
    n_added = 0
    n_total_cited = 0
    for line in open(run_dir / "generated_answers.jsonl"):
        r = json.loads(line)
        ans = r.get("answer_text", "")
        toks = content_tokens(ans)
        cited = list(r.get("cited_chunk_ids", []))
        n_total_cited += len(cited)
        sel_for_q = sel.get(r["query_id"], [])

        # Build supports map for all selected chunks
        supports = {cid: chunk_supports(chunk_text.get(cid, ""), toks) for cid in sel_for_q}

        # 1. Replace citations with no support if a better chunk exists
        new_cited: list[str] = []
        for cid in cited:
            if supports.get(cid, 0) >= 1:
                new_cited.append(cid)
            else:
                # Find replacement from selected_context with highest support
                best = max(sel_for_q, key=lambda x: supports.get(x, 0)) if sel_for_q else None
                if best is not None and supports.get(best, 0) >= 1 and best not in new_cited:
                    new_cited.append(best)
                    n_replaced += 1
                else:
                    new_cited.append(cid)  # leave as-is if no better option

        # 2. If no citations made it but a supported chunk exists, add one
        if not new_cited and toks:
            best = max(sel_for_q, key=lambda x: supports.get(x, 0)) if sel_for_q else None
            if best is not None and supports.get(best, 0) >= 1:
                new_cited.append(best)
                n_added += 1

        r["cited_chunk_ids_orig"] = cited
        r["cited_chunk_ids"] = new_cited
        r["cpv_changed"] = cited != new_cited
        out_rows.append(r)

    with open(run_dir / args.out_name, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Recompute citation_accuracy if queries provided
    if args.queries:
        gold = {json.loads(l)["query_id"]: set(json.loads(l)["gold_doc_ids"])
                for l in open(args.queries)}
        n = len(gold)
        cit_orig = 0.0
        cit_cpv = 0.0
        for r in out_rows:
            qid = r["query_id"]
            gd = gold.get(qid, set())
            o = r["cited_chunk_ids_orig"]
            c = r["cited_chunk_ids"]
            if o:
                cit_orig += sum(1 for cid in o if chunk_doc.get(cid) in gd) / len(o)
            if c:
                cit_cpv += sum(1 for cid in c if chunk_doc.get(cid) in gd) / len(c)
        print(f"n={n}  replaced={n_replaced}  added={n_added}  total_cited_orig={n_total_cited}")
        print(f"cit_acc_orig={cit_orig/n:.4f}")
        print(f"cit_acc_cpv ={cit_cpv/n:.4f}")
        print(f"delta       =+{(cit_cpv-cit_orig)/n*100:.2f}pp")


if __name__ == "__main__":
    main()
