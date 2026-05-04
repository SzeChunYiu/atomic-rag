"""CPU-only: BM25 re-rank chunks within already-retrieved candidates.

Targets S2 (selection_chunk_miss): the gold doc is retrieved but the wrong
chunk from that doc gets selected. We re-score all candidate chunks from
the top-N retrieved docs using BM25 against the query, then reselect.

This changes the selected_context without re-running generation — we produce
a new selected_context.jsonl and measure expected coverage improvement.

Usage:
    python scripts/rerank_chunks_bm25.py \
        --run-dir RUNDIR \
        --queries QUERIES_JSONL \
        --chunks  CHUNKS_JSONL \
        --out-dir OUTDIR
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")

K1 = 1.5
B = 0.75


def tokenize(s: str) -> list[str]:
    s = _PUNC.sub(" ", s.lower())
    return [t for t in _WS.sub(" ", s).strip().split() if len(t) > 1]


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return _WS.sub(" ", s).strip()


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


class BM25:
    def __init__(self, docs: list[tuple[str, str]]) -> None:
        self.ids = [d[0] for d in docs]
        self.corpus = [tokenize(d[1]) for d in docs]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(1, self.N)
        df: Counter = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        self.idf = {
            t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def score(self, query: str, idx: int) -> float:
        q_toks = tokenize(query)
        doc = self.corpus[idx]
        dl = len(doc)
        tf_map = Counter(doc)
        s = 0.0
        for t in q_toks:
            if t not in self.idf:
                continue
            tf = tf_map.get(t, 0)
            num = tf * (K1 + 1)
            den = tf + K1 * (1 - B + B * dl / self.avgdl)
            s += self.idf[t] * num / den
        return s

    def rank(self, query: str) -> list[tuple[str, float]]:
        scored = [(self.ids[i], self.score(query, i)) for i in range(self.N)]
        return sorted(scored, key=lambda x: -x[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--token-budget", type=int, default=1024)
    args = ap.parse_args()

    # Load data
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

    # Load candidates (full retrieved pool)
    cands_by_q: dict[str, list[str]] = defaultdict(list)
    cand_path = args.run_dir / "candidates.jsonl"
    if cand_path.is_file():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cands_by_q[row["query_id"]].append(row["chunk_id"])

    # Load original selection
    orig_sel_by_q: dict[str, list[str]] = defaultdict(list)
    sel_path = args.run_dir / "selected_context.jsonl"
    if sel_path.is_file():
        for line in sel_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            orig_sel_by_q[row["query_id"]].append(row["chunk_id"])

    # Compare coverage: original vs BM25-reranked selection
    orig_gold_text_hits = new_gold_text_hits = 0
    orig_f1_sum = new_f1_sum = 0.0
    total = 0

    def select_by_budget(ranked_cids: list[str], budget: int) -> list[str]:
        out = []
        used = 0
        for cid in ranked_cids:
            t = len(tokenize(chunk_text.get(cid, "")))
            if used + t <= budget:
                out.append(cid)
                used += t
        return out

    def content_tokens(s: str) -> set[str]:
        stop = set("a an the of in on at to for from with by is are was were be been "
                   "being and or but if then that this these those it its".split())
        return {t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b", s)
                if t.lower() not in stop}

    new_selections = []

    for qid, q in queries.items():
        gold_docs = set(q.get("gold_doc_ids") or [])
        gold_ans = q.get("metadata", {}).get("answer") or ""
        refs = [gold_ans] if gold_ans else []
        gold_tok = content_tokens(gold_ans)

        query_text = q.get("text", "")
        candidates = cands_by_q.get(qid, [])
        if not candidates:
            candidates = orig_sel_by_q.get(qid, [])

        # BM25 re-rank all candidate chunks
        docs_to_score = [(cid, chunk_text.get(cid, "")) for cid in candidates]
        if not docs_to_score:
            continue

        bm25 = BM25(docs_to_score)
        ranked = bm25.rank(query_text)
        ranked_cids = [cid for cid, _ in ranked]

        new_sel = select_by_budget(ranked_cids, args.token_budget)
        orig_sel = orig_sel_by_q.get(qid, [])

        # Measure gold text coverage
        orig_text = " ".join(chunk_text.get(c, "") for c in orig_sel)
        new_text = " ".join(chunk_text.get(c, "") for c in new_sel)

        orig_hit = bool(gold_tok) and gold_tok.issubset(content_tokens(orig_text))
        new_hit = bool(gold_tok) and gold_tok.issubset(content_tokens(new_text))

        orig_gold_text_hits += int(orig_hit)
        new_gold_text_hits += int(new_hit)
        total += 1

        new_selections.append({
            "query_id": qid,
            "new_selected": new_sel,
            "orig_selected": orig_sel,
            "orig_gold_text": orig_hit,
            "new_gold_text": new_hit,
        })

    print(f"\n=== BM25 chunk re-ranking coverage analysis ===")
    print(f"Total queries: {total}")
    print(f"Gold answer text in context:")
    print(f"  Original (SNR-greedy):  {orig_gold_text_hits}/{total} ({orig_gold_text_hits/total:.1%})")
    print(f"  BM25 re-ranked:         {new_gold_text_hits}/{total} ({new_gold_text_hits/total:.1%})")
    print(f"  Delta:                  {(new_gold_text_hits - orig_gold_text_hits)/total:+.1%}")

    improved = sum(1 for r in new_selections if r["new_gold_text"] and not r["orig_gold_text"])
    worsened = sum(1 for r in new_selections if r["orig_gold_text"] and not r["new_gold_text"])
    print(f"  Improved by BM25: {improved}  Worsened: {worsened}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bm25_rerank_analysis.jsonl").write_text(
        "\n".join(json.dumps(r) for r in new_selections) + "\n", encoding="utf-8"
    )
    (args.out_dir / "coverage_summary.json").write_text(
        json.dumps({
            "orig_gold_text_coverage": orig_gold_text_hits / total,
            "bm25_gold_text_coverage": new_gold_text_hits / total,
            "delta": (new_gold_text_hits - orig_gold_text_hits) / total,
            "improved": improved,
            "worsened": worsened,
            "total": total,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nAnalysis written to {args.out_dir}")


if __name__ == "__main__":
    main()
