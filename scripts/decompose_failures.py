"""Fine-grained failure decomposition for the three RAG stages.

For each query, classifies the failure into the smallest possible sub-cause:

RETRIEVE failures:
  R1 - gold doc not in top-50 (hard miss)

SELECT failures (given gold retrieved):
  S1 - gold doc retrieved but no chunk selected from it
  S2 - gold chunk selected but answer text not present in it
  S3 - bridge incomplete: one gold doc selected, other missing (multi-hop specific)

GENERATE failures (given correct evidence selected):
  G1 - answer token overlap is zero (complete confabulation)
  G2 - partial overlap but wrong final answer (reasoning error)
  G3 - "I don't know" / abstention
  G4 - correct answer buried in verbose output (extractable by tighter parsing)

Prints a breakdown table and per-query CSV.

Usage:
    python scripts/decompose_failures.py \
        --run-dir runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/736099aaa5bf \
        --queries data/hotpotqa_1k/queries.jsonl \
        --chunks  runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/index_bundle/chunks.jsonl \
        --candidates runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/736099aaa5bf/candidates.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path

_CITE_RE = re.compile(r"\[E\d+\]")
_FINAL_RE = re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE)
_IDONOTKNOW_RE = re.compile(r"i don.?t know|cannot determine|not enough|unclear", re.IGNORECASE)
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_STOP = set(
    "a an the of in on at to for from with by is are was were be been being "
    "and or but if then so than that this these those it its as which who".split()
)
_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b")


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def extract(s: str) -> str:
    m = _FINAL_RE.search(s)
    return norm(_CITE_RE.sub("", m.group(1) if m else s).strip())


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def token_f1(pred: str, refs: list[str]) -> float:
    pt = norm(pred).split()
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
        p, rec = nc / len(pt), nc / len(rt)
        best = max(best, 2 * p * rec / (p + rec))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    # Load gold
    queries: dict[str, dict] = {}
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        queries[q["query_id"]] = q

    # Load chunk → doc, chunk text
    chunk_doc: dict[str, str] = {}
    chunk_text: dict[str, str] = {}
    for line in args.chunks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        chunk_doc[c["chunk_id"]] = c["doc_id"]
        chunk_text[c["chunk_id"]] = c.get("text", "")

    # Load retrieved candidates per query (top-50 pool)
    retrieved_docs: dict[str, set[str]] = {}
    cand_path = args.candidates or (args.run_dir / "candidates.jsonl")
    if cand_path.is_file():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["query_id"]
            doc_id = chunk_doc.get(row["chunk_id"], "")
            retrieved_docs.setdefault(qid, set()).add(doc_id)

    # Load selected context
    sel_by_q: dict[str, list[str]] = {}
    sel_path = args.run_dir / "selected_context.jsonl"
    if sel_path.is_file():
        for line in sel_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sel_by_q.setdefault(row["query_id"], []).append(row["chunk_id"])

    # Load generated answers
    answers: list[dict] = []
    for line in (args.run_dir / "generated_answers.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        answers.append(json.loads(line))

    # --- Classify each query ---
    rows_out = []
    counts: Counter = Counter()
    f1_by_class: dict[str, list[float]] = {}

    for row in answers:
        qid = row["query_id"]
        q = queries.get(qid)
        if q is None:
            continue

        gold_docs = set(q.get("gold_doc_ids") or [])
        gold_ans = q.get("metadata", {}).get("answer") or ""
        refs = [gold_ans] if gold_ans else []
        gold_tok = content_tokens(gold_ans)

        pred_raw = row.get("answer_text", "")
        pred = extract(pred_raw)
        f1 = token_f1(pred, refs)
        success = f1 >= args.f1_threshold

        retrieved = retrieved_docs.get(qid, set())
        selected_cids = sel_by_q.get(qid) or row.get("selected_chunk_ids") or []
        selected_docs = {chunk_doc[c] for c in selected_cids if c in chunk_doc}
        selected_text = " ".join(chunk_text.get(c, "") for c in selected_cids)

        gold_in_retrieved = bool(gold_docs & retrieved) if retrieved else None
        gold_in_selected = bool(gold_docs & selected_docs)
        gold_text_in_selected = bool(gold_tok) and gold_tok.issubset(content_tokens(selected_text))
        bridge_complete = gold_docs.issubset(selected_docs) if len(gold_docs) >= 2 else None

        # Classify failure
        if success:
            fail_class = "SUCCESS"
        elif gold_in_retrieved is False:
            fail_class = "R1_retrieval_hard_miss"
        elif not gold_in_selected:
            if bridge_complete is False:
                fail_class = "S3_bridge_incomplete"
            else:
                fail_class = "S1_selection_doc_miss"
        elif not gold_text_in_selected:
            fail_class = "S2_selection_chunk_miss"
        else:
            # Generation failure — sub-classify
            if _IDONOTKNOW_RE.search(pred_raw):
                fail_class = "G3_abstention"
            elif token_f1(pred, refs) == 0.0:
                fail_class = "G1_confabulation"
            else:
                # Check if answer is present in raw output but extraction missed it
                alt_pred = norm(_CITE_RE.sub("", pred_raw))
                alt_f1 = token_f1(alt_pred, refs)
                if alt_f1 >= args.f1_threshold:
                    fail_class = "G4_extraction_error"
                else:
                    fail_class = "G2_reasoning_error"

        counts[fail_class] += 1
        f1_by_class.setdefault(fail_class, []).append(f1)
        rows_out.append({
            "query_id": qid,
            "query": q.get("text", "")[:80],
            "gold_ans": gold_ans,
            "pred": pred,
            "f1": round(f1, 3),
            "fail_class": fail_class,
            "bridge_complete": bridge_complete,
            "gold_in_selected": gold_in_selected,
            "gold_text_in_sel": gold_text_in_selected,
        })

    total = len(rows_out)
    print(f"\n=== Fine-grained failure decomposition ({args.run_dir.name}) ===")
    print(f"Total queries: {total}")
    print(f"\n{'Class':<35} {'N':>5} {'%':>6}  {'Mean F1':>8}  Description")
    print("-" * 75)

    order = ["SUCCESS",
             "R1_retrieval_hard_miss",
             "S1_selection_doc_miss", "S2_selection_chunk_miss", "S3_bridge_incomplete",
             "G1_confabulation", "G2_reasoning_error", "G3_abstention", "G4_extraction_error"]

    descriptions = {
        "SUCCESS": "F1 >= threshold",
        "R1_retrieval_hard_miss": "Gold doc not in top-50",
        "S1_selection_doc_miss": "Gold doc retrieved but not selected",
        "S2_selection_chunk_miss": "Gold doc selected, answer text missing",
        "S3_bridge_incomplete": "One hop selected, other missing",
        "G1_confabulation": "Zero token overlap with gold (complete hallucination)",
        "G2_reasoning_error": "Partial overlap — model can't chain the evidence",
        "G3_abstention": "'I don't know' despite gold being present",
        "G4_extraction_error": "Answer present in full output, extraction failed",
    }

    for cls in order:
        n = counts.get(cls, 0)
        if n == 0:
            continue
        mean_f1 = sum(f1_by_class.get(cls, [0])) / max(1, len(f1_by_class.get(cls, [])))
        print(f"  {cls:<33} {n:>5} {n/total:>6.1%}  {mean_f1:>8.3f}  {descriptions.get(cls,'')}")

    print(f"\nFix priority (by query count):")
    for cls, n in counts.most_common():
        if cls == "SUCCESS":
            continue
        print(f"  {cls}: {n} queries ({n/total:.1%})")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nPer-query CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
