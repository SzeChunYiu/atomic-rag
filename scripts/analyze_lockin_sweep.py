"""Compare lock-in retrieval vs fusion_rrf baseline.

Reports retrieval-side metrics (recall@k, MRR) and selection-side metrics
(citation_accuracy, answer_f1) at cs=384. Lock-in's theoretical claim is
~√M SNR boost on invariant evidence — should manifest as recall@5 ↑.

Usage:
    python scripts/analyze_lockin_sweep.py \
        --root /projects/hep/fs10/shared/nnbar/billy/RAG/runs/p4_lockin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(d: Path) -> dict:
    files = list(d.glob("*/metrics.json"))
    return json.loads(files[0].read_text()) if files else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    configs = [
        ("fusion_rrf (baseline)", "hotpotqa_1k_cs384_fusion_rrf_baseline"),
        ("lock-in n=4", "hotpotqa_1k_cs384_lockin_n4"),
    ]
    rows = []
    for label, name in configs:
        m = load_metrics(args.root / name)
        if not m:
            print(f"  missing: {name}")
            continue
        rows.append({
            "config": label,
            "recall@1": m.get("recall@1_doc_mean"),
            "recall@5": m.get("recall@5_doc_mean"),
            "recall@10": m.get("recall@10_doc_mean"),
            "MRR": m.get("mean_reciprocal_rank_doc"),
            "ndcg@5": m.get("ndcg@5_chunk_mean"),
            "cit_acc": m.get("citation_accuracy_mean"),
            "f1": m.get("answer_f1_mean"),
            "retrieve_seconds": m.get("retrieve_seconds"),
        })

    if len(rows) < 2:
        print("(need both configs; aborting)")
        return

    print("\n=== retrieval-side comparison ===")
    print(f"{'metric':<20}  {'fusion_rrf':>12}  {'lockin n=4':>12}  {'Δ':>10}")
    for k in ("recall@1", "recall@5", "recall@10", "MRR", "ndcg@5", "cit_acc", "f1"):
        a, b = rows[0][k], rows[1][k]
        if a is None or b is None:
            continue
        delta = b - a
        rel = (delta / a * 100) if a != 0 else 0
        print(f"{k:<20}  {a:>12.4f}  {b:>12.4f}  {delta:>+10.4f} ({rel:+.1f}%)")

    print(f"\nretrieve_seconds: fusion={rows[0]['retrieve_seconds']:.1f}s  lockin={rows[1]['retrieve_seconds']:.1f}s")


if __name__ == "__main__":
    main()
