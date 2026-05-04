"""Extract final-answer text + citations from a CoT-prompted run.

Reads `generated_answers.jsonl` from a run dir; produces
`generated_answers_cot.jsonl` where:
- `answer_text` = the substring AFTER the last "Final answer:" marker
- `cited_chunk_ids` = `[E_i]` markers found ONLY on the final-answer line

Recomputes citation_accuracy + a token-F1 if --queries provided.
"""
from __future__ import annotations
import argparse, json, re, string
from collections import Counter
from pathlib import Path

CIT_RE = re.compile(r"\[E(\d+)\]")
PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
ART = re.compile(r"\b(a|an|the)\b")
WS = re.compile(r"\s+")

def normalize(s):
    s = PUNC.sub(" ", s.lower())
    s = ART.sub(" ", s)
    return WS.sub(" ", s).strip()

def f1(pred, refs):
    if not refs: return 0.0
    pt = normalize(pred).split()
    if not pt: return 0.0
    best = 0.0
    for r in refs:
        rt = normalize(r).split()
        if not rt: continue
        common = Counter(pt) & Counter(rt)
        n = sum(common.values())
        if n == 0: continue
        p = n/len(pt); rec = n/len(rt)
        best = max(best, 2*p*rec/(p+rec))
    return best


def extract_final(text):
    """Return (final_answer_text, citations_on_final_line)."""
    m = list(re.finditer(r"final\s*answer\s*:\s*", text, flags=re.IGNORECASE))
    if not m:
        # No "Final answer:" found — fall back to last non-empty line.
        line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), text)
    else:
        line = text[m[-1].end():]
        line = line.split("\n", 1)[0]
    cits = [int(x) for x in CIT_RE.findall(line)]
    final_only = CIT_RE.sub(" ", line).strip()
    return final_only, cits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--out_name", default="generated_answers_cot.jsonl")
    args = ap.parse_args()

    run_dir = args.run_dir
    parent = run_dir.parent
    chunk_doc = {}
    for line in open(parent / "index_bundle" / "chunks.jsonl"):
        c = json.loads(line); chunk_doc[c["chunk_id"]] = c["doc_id"]

    rows = []
    for line in open(run_dir / "generated_answers.jsonl"):
        r = json.loads(line)
        full = r.get("answer_text", "")
        sel_ids = r.get("selected_chunk_ids", [])
        final, cits = extract_final(full)
        cit_ids = []
        seen = set()
        for i in cits:
            if 1 <= i <= len(sel_ids):
                cid = sel_ids[i-1]
                if cid not in seen:
                    cit_ids.append(cid); seen.add(cid)
        r["answer_text_full"] = full
        r["answer_text"] = final
        r["cited_chunk_ids_orig"] = r.get("cited_chunk_ids", [])
        r["cited_chunk_ids"] = cit_ids
        rows.append(r)

    with open(run_dir / args.out_name, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.queries:
        gold = {}
        for line in open(args.queries):
            q = json.loads(line)
            ans = q.get("metadata", {}).get("answer")
            gold[q["query_id"]] = (
                set(q["gold_doc_ids"]),
                [ans] if isinstance(ans, str) else (ans or []),
            )
        n = len(gold)
        cit = 0.0
        f1_full = 0.0
        f1_final = 0.0
        em_final = 0.0
        em_full = 0.0
        for r in rows:
            qid = r["query_id"]
            gd, ans_gold = gold.get(qid, (set(), []))
            cits = r["cited_chunk_ids"]
            if cits:
                cit += sum(1 for cid in cits if chunk_doc.get(cid) in gd) / len(cits)
            f1_full += f1(r["answer_text_full"], ans_gold)
            f1_final += f1(r["answer_text"], ans_gold)
            pred_norm = normalize(r["answer_text"])
            em_final += float(any(pred_norm == normalize(a) for a in ans_gold))
            full_norm = normalize(r["answer_text_full"])
            em_full += float(any(full_norm == normalize(a) for a in ans_gold))
        print(f"n={n}")
        print(f"cit_acc_cot         ={cit/n:.4f}")
        print(f"F1_on_full_response ={f1_full/n:.4f}  EM={em_full/n:.4f}")
        print(f"F1_on_final_only    ={f1_final/n:.4f}  EM={em_final/n:.4f}")


if __name__ == "__main__":
    main()
