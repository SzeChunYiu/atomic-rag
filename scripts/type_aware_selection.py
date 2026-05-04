"""CPU-only answer-type-aware chunk selection supplement.

For multi-hop questions where the dense retriever picks the wrong paragraph
within the correct document, this re-scores candidate chunks by whether they
contain content likely to answer the detected question type.

Strategy:
  1. Detect answer type from question (nationality, place, person, date, boolean)
  2. Score each candidate chunk by answer-type presence (regex patterns)
  3. For each already-selected document, swap in its highest type-score chunk
     if it isn't already selected — filling the gap without touching budget.

This directly targets S2 failures: gold doc selected, answer text absent.
CPU-only: no embeddings recomputed, no model calls.

Usage:
    python scripts/type_aware_selection.py \
        --run-dir RUNDIR \
        --queries QUERIES \
        --chunks  CHUNKS \
        --out-dir OUTDIR \
        [--measure-only]   # just report coverage delta, don't write
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")

# Answer-type detectors (question → type)
_Q_TYPE = [
    (re.compile(r"\bnationality\b|\bcitizenship\b|\bcountry.of.origin\b", re.I), "nationality"),
    (re.compile(r"\bborn.in\b|\bbirth.?place\b|\bnative.of\b|\bplace.of.birth\b", re.I), "place"),
    (re.compile(r"\bplace.of.death\b|\bdied.in\b|\bplace.of.burial\b|\bburied\b", re.I), "place"),
    (re.compile(r"\bdirected.by\b|\bwho.directed\b|\bdirector\b", re.I), "person"),
    (re.compile(r"\bwrote\b|\bauthor.of\b|\bwritten.by\b|\bcomposed.by\b", re.I), "person"),
    (re.compile(r"\bwho.starred\b|\bactor\b|\bactress\b|\bplayed.by\b", re.I), "person"),
    (re.compile(r"\bsibling\b|\bspouse\b|\bhusband\b|\bwife\b|\bmother\b|\bfather\b|\bson\b|\bdaughter\b", re.I), "person"),
    (re.compile(r"\bwhen\b|\byear\b|\bfounded\b|\bestablished\b|\bborn\b.*\bwhen\b", re.I), "date"),
    (re.compile(r"\bare.*same\b|\bboth.*located\b|\bboth.*from\b|\bin.common\b", re.I), "boolean"),
    (re.compile(r"\bwhere\b|\blocation\b|\bplace\b|\bcountry\b|\bcity\b", re.I), "place"),
    (re.compile(r"\bwho\b", re.I), "person"),
]

# Type → chunk scoring patterns
_NATIONALITIES = set("""american british french german italian spanish portuguese dutch swedish
norwegian danish finnish russian chinese japanese korean indian australian canadian
brazilian mexican argentinian greek turkish polish czech hungarian romanian ukrainian
iranian egyptian moroccan thai indonesian philippine vietnamese swiss austrian belgian
south african nigerian ghanaian icelandic estonian latvian lithuanian""".split())

_PLACE_PATTERNS = [
    re.compile(r"\bin ([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b"),
    re.compile(r"\bat ([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b"),
    re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2}),\s*[A-Z][a-z]+"),
]

_DATE_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")

_PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")

_BOOL_PATTERNS = [re.compile(r"\byes\b|\bno\b|\bboth\b|\bneither\b|\bsame\b", re.I)]


def detect_type(query: str) -> str:
    for pat, qtype in _Q_TYPE:
        if pat.search(query):
            return qtype
    return "general"


def score_chunk_for_type(text: str, qtype: str) -> float:
    """Score chunk 0-1 by how likely it contains an answer of qtype."""
    text_lower = text.lower()
    if qtype == "nationality":
        hits = sum(1 for nat in _NATIONALITIES if nat in text_lower)
        return min(1.0, hits * 0.3)
    if qtype == "place":
        hits = sum(len(p.findall(text)) for p in _PLACE_PATTERNS)
        return min(1.0, hits * 0.15)
    if qtype == "date":
        years = _DATE_PATTERN.findall(text)
        return min(1.0, len(years) * 0.25)
    if qtype == "person":
        persons = _PERSON_PATTERN.findall(text)
        return min(1.0, len(persons) * 0.1)
    if qtype == "boolean":
        hits = sum(len(p.findall(text)) for p in _BOOL_PATTERNS)
        return min(1.0, hits * 0.2)
    return 0.0


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return _WS.sub(" ", s).strip()


def content_tokens(s: str) -> set[str]:
    stop = set("a an the of in on at to for from with by is are was were be been "
               "being and or but if then that this these those it its".split())
    return {t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z]{1,}\b", s)
            if t.lower() not in stop}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--token-budget", type=int, default=1024)
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args()

    # Load
    queries: dict[str, dict] = {}
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        queries[q["query_id"]] = q

    chunk_text: dict[str, str] = {}
    chunk_doc: dict[str, str] = {}
    doc_chunks: dict[str, list[str]] = defaultdict(list)
    for line in args.chunks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        chunk_text[c["chunk_id"]] = c.get("text", "")
        chunk_doc[c["chunk_id"]] = c["doc_id"]
        doc_chunks[c["doc_id"]].append(c["chunk_id"])

    # Load candidate pool (top-50)
    cands_by_q: dict[str, list[str]] = defaultdict(list)
    cand_path = args.run_dir / "candidates.jsonl"
    if cand_path.is_file():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cands_by_q[row["query_id"]].append(row["chunk_id"])

    # Load original selection
    orig_sel: dict[str, list[str]] = defaultdict(list)
    sel_path = args.run_dir / "selected_context.jsonl"
    if sel_path.is_file():
        for line in sel_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            orig_sel[row["query_id"]].append(row["chunk_id"])

    orig_hits = type_hits = 0
    improved = worsened = 0
    new_selections: list[dict] = []

    for qid, q in queries.items():
        gold_tok = content_tokens(q.get("metadata", {}).get("answer") or "")
        query_text = q.get("text", "")
        qtype = detect_type(query_text)

        selected = list(orig_sel.get(qid, []))
        candidates = cands_by_q.get(qid, selected)

        selected_docs = {chunk_doc.get(c, "") for c in selected}
        used_tokens = sum(len(chunk_text.get(c, "").split()) for c in selected)

        # For each already-selected doc, find highest type-score chunk not yet selected
        for doc_id in selected_docs:
            doc_cids = [c for c in doc_chunks.get(doc_id, [])
                        if c in {cc for cc in candidates} and c not in selected]
            if not doc_cids:
                continue
            best = max(doc_cids, key=lambda c: score_chunk_for_type(chunk_text.get(c, ""), qtype))
            best_score = score_chunk_for_type(chunk_text.get(best, ""), qtype)
            if best_score < 0.1:
                continue
            extra_tokens = len(chunk_text.get(best, "").split())
            if used_tokens + extra_tokens <= args.token_budget:
                selected.append(best)
                used_tokens += extra_tokens

        orig_text = " ".join(chunk_text.get(c, "") for c in orig_sel.get(qid, []))
        new_text = " ".join(chunk_text.get(c, "") for c in selected)

        orig_hit = bool(gold_tok) and gold_tok.issubset(content_tokens(orig_text))
        new_hit = bool(gold_tok) and gold_tok.issubset(content_tokens(new_text))
        orig_hits += int(orig_hit)
        type_hits += int(new_hit)
        if new_hit and not orig_hit:
            improved += 1
        elif orig_hit and not new_hit:
            worsened += 1

        new_selections.append({"query_id": qid, "chunk_ids": selected, "answer_type": qtype})

    total = len(new_selections)
    print(f"\n=== Type-aware selection supplement ===")
    print(f"Total queries: {total}")
    print(f"Gold text coverage:")
    print(f"  Original:   {orig_hits}/{total} ({orig_hits/total:.1%})")
    print(f"  Type-aware: {type_hits}/{total} ({type_hits/total:.1%})")
    print(f"  Delta:      {(type_hits-orig_hits)/total:+.1%}")
    print(f"  Improved: {improved}  Worsened: {worsened}")

    if not args.measure_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "type_aware_selection.jsonl").write_text(
            "\n".join(json.dumps(r) for r in new_selections) + "\n", encoding="utf-8"
        )
        (args.out_dir / "coverage_summary.json").write_text(
            json.dumps({
                "orig_coverage": orig_hits / total,
                "type_aware_coverage": type_hits / total,
                "delta": (type_hits - orig_hits) / total,
                "improved": improved, "worsened": worsened,
            }, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
