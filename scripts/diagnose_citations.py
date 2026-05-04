"""Citation-failure-mode diagnostic.

For each generated answer, classify its citations into three buckets:
1. supported & correct doc — gold-doc + token-overlap-with-answer >= 1
2. supported but wrong doc — token-overlap >= 1 but doc not in gold
3. hallucinated — token-overlap == 0 with the answer

Drives the choice of citation-cleanup method:
- bucket 2 large -> retrieval/selection problem (atom is in wrong doc)
- bucket 3 large -> generator hallucinating cite ids (NLI/rollout fix)
- bucket 1 large -> citations are mostly fine; cit_acc gap is metric/normalization
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter


_STOP = set("a an the of in on at to for from with by is are was were be been "
            "being and or but if then so than that this these those it its "
            "as which who whom whose what when where why how".split())
_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="dir with generated_answers.jsonl + selected_atoms.jsonl")
    ap.add_argument("--queries", type=Path, required=True)
    args = ap.parse_args()

    queries = {q["query_id"]: q
               for q in (json.loads(l) for l in open(args.queries))}
    atoms_by_q: dict[str, dict[str, dict]] = {}
    for line in open(args.run_dir / "selected_atoms.jsonl"):
        s = json.loads(line)
        atoms_by_q.setdefault(s["query_id"], {})[s["atom_id"]] = s
    answers = [json.loads(l)
               for l in open(args.run_dir / "generated_answers.jsonl")]

    n_queries = len(answers)
    n_total = n_supported_correct = n_supported_wrong = n_hallucinated = 0
    n_uncited = 0
    overlap_supported = []
    for r in answers:
        qid = r["query_id"]
        cited = r.get("cited_atom_ids") or []
        if not cited:
            n_uncited += 1
            continue
        ans_tok = content_tokens(r.get("answer_text") or "")
        if not ans_tok:
            continue
        gold_docs = set(queries[qid]["gold_doc_ids"])
        sel = atoms_by_q.get(qid, {})
        for aid in cited:
            n_total += 1
            atom = sel.get(aid)
            if atom is None:
                n_hallucinated += 1
                continue
            atom_tok = content_tokens(atom["text"])
            overlap = len(ans_tok & atom_tok)
            if overlap == 0:
                n_hallucinated += 1
            elif atom["doc_id"] in gold_docs:
                n_supported_correct += 1
                overlap_supported.append(overlap)
            else:
                n_supported_wrong += 1
                overlap_supported.append(overlap)

    print(f"queries: {n_queries}  uncited_queries: {n_uncited}")
    print(f"citations: {n_total}")
    if n_total > 0:
        print(f"  supported & correct doc: {n_supported_correct} "
              f"({n_supported_correct/n_total:.1%})")
        print(f"  supported but wrong doc: {n_supported_wrong} "
              f"({n_supported_wrong/n_total:.1%})")
        print(f"  hallucinated (no overlap): {n_hallucinated} "
              f"({n_hallucinated/n_total:.1%})")
    if overlap_supported:
        c = Counter(overlap_supported)
        med = sorted(overlap_supported)[len(overlap_supported)//2]
        print(f"overlap (supported only): mean={sum(overlap_supported)/len(overlap_supported):.2f} "
              f"median={med}  histogram (top 5): {c.most_common(5)}")


if __name__ == "__main__":
    main()
