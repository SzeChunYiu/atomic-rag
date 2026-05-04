"""Post-hoc citation cleanup.

Diagnostic showed 55% of D04+D06 citations have zero token overlap with
the generated answer (hallucinated IDs, not noisy attribution). This
script rewrites citations: for each cited atom_id with zero answer-text
overlap, replace with the highest-overlap atom from the selected pool
(or drop if no atom in pool overlaps).

Reads generated_answers.jsonl + selected_atoms.jsonl, writes
generated_answers_cleaned.jsonl + metrics_cleaned.json.

This is the trivial-baseline fix for citation hallucination — sets the
floor that any more-sophisticated method (NLI, attention rollout) must
beat. Predicted: cit_acc 0.55 -> 0.85+, F1 unchanged.
"""
from __future__ import annotations
import argparse, json, re, string
from pathlib import Path
from collections import Counter


_STOP = set("a an the of in on at to for from with by is are was were be been "
            "being and or but if then so than that this these those it its as "
            "which who whom whose what when where why how".split())
_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_EVID = re.compile(r"\[E\d+\]")


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def normalize(s):
    s = _EVID.sub(" ", s); s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s); return _WS.sub(" ", s).strip()


def tokf1(p, refs):
    if not refs: return 0.0
    pt = normalize(p).split()
    if not pt: return 0.0
    best = 0.0
    for r in refs:
        rt = normalize(r).split()
        if not rt: continue
        common = Counter(pt) & Counter(rt)
        n = sum(common.values())
        if n == 0: continue
        prec = n/len(pt); rec = n/len(rt)
        best = max(best, 2*prec*rec/(prec+rec))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--mode", default="replace",
                    choices=["replace", "drop"],
                    help="replace zero-overlap with highest-overlap; or drop")
    ap.add_argument("--min_overlap", type=int, default=1)
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries))}
    sel_by_q: dict[str, list[dict]] = {}
    for line in open(args.run_dir / "selected_atoms.jsonl"):
        s = json.loads(line)
        sel_by_q.setdefault(s["query_id"], []).append(s)

    answers = [json.loads(l)
               for l in open(args.run_dir / "generated_answers.jsonl")]

    out_rows = []
    cit_sum_orig = cit_sum_clean = 0.0
    n_orig_cit = n_clean_cit = 0
    n_replaced = n_dropped = 0

    for r in answers:
        qid = r["query_id"]
        ans_tok = content_tokens(r.get("answer_text") or "")
        gold_docs = set(queries[qid]["gold_doc_ids"])
        pool = sel_by_q.get(qid, [])
        atom_by_id = {a["atom_id"]: a for a in pool}

        # Original cit_acc
        orig_cited = r.get("cited_atom_ids") or []
        if orig_cited:
            n_hit = sum(1 for aid in orig_cited
                        if atom_by_id.get(aid, {}).get("doc_id") in gold_docs)
            cit_sum_orig += n_hit / len(orig_cited)
            n_orig_cit += 1

        # Clean: per citation, check overlap; if zero, replace or drop
        clean_cited = []
        for aid in orig_cited:
            atom = atom_by_id.get(aid)
            overlap = len(ans_tok & content_tokens(atom["text"])) if atom else 0
            if overlap >= args.min_overlap:
                clean_cited.append(aid)
            else:
                if args.mode == "replace" and ans_tok:
                    best = None; best_ov = 0
                    for cand in pool:
                        ov = len(ans_tok & content_tokens(cand["text"]))
                        if ov > best_ov:
                            best_ov = ov; best = cand
                    if best is not None and best_ov >= args.min_overlap:
                        clean_cited.append(best["atom_id"])
                        n_replaced += 1
                    else:
                        n_dropped += 1
                else:
                    n_dropped += 1
        if clean_cited:
            n_hit = sum(1 for aid in clean_cited
                        if atom_by_id.get(aid, {}).get("doc_id") in gold_docs)
            cit_sum_clean += n_hit / len(clean_cited)
            n_clean_cit += 1

        out_rows.append({**r, "cited_atom_ids_clean": clean_cited})

    with open(args.run_dir / "generated_answers_cleaned.jsonl", "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cleaned_metrics = {
        "n": len(answers),
        "cit_acc_original": cit_sum_orig / max(1, n_orig_cit),
        "cit_acc_cleaned": cit_sum_clean / max(1, n_clean_cit),
        "n_replaced": n_replaced, "n_dropped": n_dropped,
        "mode": args.mode, "min_overlap": args.min_overlap,
    }
    with open(args.run_dir / "metrics_cleaned.json", "w") as f:
        json.dump(cleaned_metrics, f, indent=2)
    print(json.dumps(cleaned_metrics, indent=2))


if __name__ == "__main__":
    main()
