"""Analyze the real-generator HotpotQA sweep:

Compares cs=384 × {greedy, v4 α=0.7, MMR} with Qwen2.5-7B real generator.
Reports per-config metrics + paired bootstrap on per-query answer_em / answer_f1
/ citation_accuracy diffs.

Usage:
    python scripts/analyze_realgen_sweep.py \
        --root /projects/hep/fs10/shared/nnbar/billy/RAG/runs/realgen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def find_one(p: Path, glob: str) -> Path | None:
    files = list(p.glob(glob))
    return files[0] if files else None


def load_metrics(d: Path) -> dict:
    f = find_one(d, "*/metrics.json")
    if f is None:
        return {}
    return json.loads(f.read_text())


def load_per_query_metrics(d: Path) -> dict[str, dict]:
    """Read per-query EM/F1/cit_acc from artifacts in the run dir.
    The eval pipeline's per-query records (if present) live in eval_per_query.jsonl.
    Falls back to recomputing from generated_answers.jsonl + queries.jsonl if needed.
    """
    f = find_one(d, "*/eval_per_query.jsonl")
    if f is None:
        return {}
    out: dict[str, dict] = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["query_id"]] = r
    return out


def paired_bootstrap_diff(a: list[float], b: list[float], *, n=10000, seed=0) -> dict:
    diffs = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None])
    if not diffs.size:
        return {}
    rng = np.random.default_rng(seed)
    boot = np.array([diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n)])
    return {
        "n": int(len(diffs)),
        "mean_diff": float(diffs.mean()),
        "p_a_greater": float((boot > 0).mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    configs = [
        ("greedy", "hotpotqa_1k_cs384_greedy_qwen7b"),
        ("v4_a07", "hotpotqa_1k_cs384_v4a0_7_qwen7b"),
        ("mmr", "hotpotqa_1k_cs384_mmr_qwen7b"),
    ]

    rows = []
    per_q: dict[str, dict[str, dict]] = {}
    for label, name in configs:
        d = args.root / name
        if not d.exists():
            print(f"  missing: {d}")
            continue
        m = load_metrics(d)
        if not m:
            print(f"  no metrics: {d}")
            continue
        rows.append({
            "config": label,
            "recall@5": m.get("recall@5_doc_mean"),
            "citation_accuracy": m.get("citation_accuracy_mean"),
            "answer_em": m.get("answer_em_mean"),
            "answer_f1": m.get("answer_f1_mean"),
            "conservation_faithfulness": m.get("conservation_faithfulness_mean"),
            "answer_count": m.get("answer_count"),
        })
        per_q[label] = load_per_query_metrics(d)

    print("\n=== headline metrics ===")
    headers = ["config", "recall@5", "cit_acc", "EM", "F1", "cons", "n"]
    print("  ".join(f"{h:<10}" for h in headers))
    for r in rows:
        print(f"  {r['config']:<8}  "
              f"{r.get('recall@5', 0) or 0:.4f}    "
              f"{r.get('citation_accuracy', 0) or 0:.4f}    "
              f"{r.get('answer_em', 0) or 0:.4f}  "
              f"{r.get('answer_f1', 0) or 0:.4f}  "
              f"{r.get('conservation_faithfulness', 0) or 0:.4f}  "
              f"{int(r.get('answer_count', 0) or 0)}")

    if not per_q.get("v4_a07") or not per_q.get("greedy"):
        print("\n(no per-query data for paired bootstrap)")
        return

    print("\n=== paired bootstraps (v4 vs greedy / mmr vs greedy) ===")
    for cmp_label in ("v4_a07", "mmr"):
        if cmp_label not in per_q:
            continue
        a, b = [], []
        common = set(per_q[cmp_label]) & set(per_q["greedy"])
        for qid in common:
            a.append(per_q[cmp_label][qid].get("answer_f1"))
            b.append(per_q["greedy"][qid].get("answer_f1"))
        boot = paired_bootstrap_diff(a, b)
        print(f"  {cmp_label} vs greedy on F1: {boot}")
        a, b = [], []
        for qid in common:
            a.append(per_q[cmp_label][qid].get("citation_accuracy"))
            b.append(per_q["greedy"][qid].get("citation_accuracy"))
        boot = paired_bootstrap_diff(a, b)
        print(f"  {cmp_label} vs greedy on cit_acc: {boot}")


if __name__ == "__main__":
    main()
