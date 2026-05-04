"""Majority-vote ensemble over existing generated_answers.jsonl runs.

CPU-only. Reads N answer files, votes per query by normalized string match,
takes the plurality answer. Ties broken by run priority order (last listed wins).

Usage:
    python scripts/ensemble_answers.py \
        --queries data/hotpotqa_1k/queries.jsonl \
        --runs RUN_DIR1 RUN_DIR2 ... \
        --out-dir runs/ensemble/hotpot_all \
        [--weights 1 1 1 2 3 3]   # optional: weight per run
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

_CITE_RE = re.compile(r"\[E\d+\]")
_FINAL_ANS_RE = re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE)
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def extract(s: str) -> str:
    m = _FINAL_ANS_RE.search(s)
    return norm(_CITE_RE.sub("", m.group(1) if m else s).strip())


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


def exact_match(pred: str, refs: list[str]) -> float:
    p = norm(pred)
    return float(any(p == norm(r) for r in refs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Vote weight per run (default: 1 each)")
    args = ap.parse_args()

    weights = args.weights
    if weights is None:
        weights = [1.0] * len(args.runs)
    if len(weights) != len(args.runs):
        raise ValueError("--weights length must match --runs length")

    # Load gold answers
    gold: dict[str, list[str]] = {}
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        ans = q.get("metadata", {}).get("answer") or ""
        gold[q["query_id"]] = [ans] if isinstance(ans, str) and ans else (ans if ans else [])

    # Collect per-query answer votes
    votes: dict[str, Counter] = defaultdict(Counter)
    raw_by_run: dict[str, dict[str, str]] = {}

    for run_dir, w in zip(args.runs, weights):
        ans_path = run_dir / "generated_answers.jsonl"
        if not ans_path.is_file():
            print(f"[warn] missing {ans_path}")
            continue
        run_answers: dict[str, str] = {}
        for line in ans_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["query_id"]
            pred = extract(row.get("answer_text", ""))
            votes[qid][pred] += w
            run_answers[qid] = pred
        raw_by_run[str(run_dir)] = run_answers

    # Per-run individual F1 (for comparison)
    print("=== Individual run F1 (re-computed from extracted answers) ===")
    for run_dir, w in zip(args.runs, weights):
    	run_answers = raw_by_run.get(str(run_dir), {})
    	if not run_answers:
    		continue
    	em_s = f1_s = 0.0
    	n = len(run_answers)
    	for qid, pred in run_answers.items():
    		refs = gold.get(qid, [])
    		em_s += exact_match(pred, refs)
    		f1_s += token_f1(pred, refs)
    	label = run_dir.parent.name
    	print(f"  {label:45s} EM={em_s/n:.4f}  F1={f1_s/n:.4f}  (w={w})")

    # Ensemble vote
    results = []
    em_sum = f1_sum = 0.0
    disagreements = 0

    for qid, vote_counts in votes.items():
        refs = gold.get(qid, [])
        pred, top_count = vote_counts.most_common(1)[0]
        total_weight = sum(vote_counts.values())
        confidence = top_count / total_weight
        if len(vote_counts) > 1:
            disagreements += 1

        em = exact_match(pred, refs)
        f1 = token_f1(pred, refs)
        em_sum += em
        f1_sum += f1

        results.append({
            "query_id": qid,
            "answer_text": pred,
            "confidence": round(confidence, 3),
            "em": em,
            "f1": f1,
            "votes": dict(vote_counts),
        })

    n = len(results)
    print(f"\n=== Ensemble ({len(args.runs)} runs, {n} queries) ===")
    print(f"  Disagreement rate: {disagreements/n:.1%}")
    print(f"  EM:  {em_sum/n:.4f}")
    print(f"  F1:  {f1_sum/n:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generated_answers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
    )
    metrics = {
        "answer_em_mean": em_sum / n,
        "answer_f1_mean": f1_sum / n,
        "answer_count": float(n),
        "disagreement_rate": disagreements / n,
        "n_runs": len(args.runs),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Written to {args.out_dir}")


if __name__ == "__main__":
    main()
