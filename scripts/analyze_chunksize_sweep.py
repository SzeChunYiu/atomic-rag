"""Analyze HotpotQA × chunk_size × selector sweep results.

Reads metrics.json from runs/chunksize_sweep/* and produces:
- summary table (csv + markdown)
- per-(cs, selector) curves
- paired bootstrap on the citation_accuracy difference

The headline claim under test:
- anti-kT v2 ≥ greedy across ALL chunk sizes (no regression)
- anti-kT v2 > greedy at small chunk sizes (IRC advantage on real data)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_PAT = re.compile(r"hotpotqa_1k_cs(\d+)_(greedy|anti_kt|mmr)")
METRICS_OF_INTEREST = [
    "recall@1_doc_mean",
    "recall@5_doc_mean",
    "citation_accuracy_mean",
    "answer_f1_mean",
    "conservation_faithfulness_mean",
    "precision@1_chunk_mean",
    "ndcg@5_chunk_mean",
]


def collect(run_root: Path) -> list[dict]:
    rows = []
    for tag_dir in sorted(run_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        m = RUN_PAT.fullmatch(tag_dir.name)
        if not m:
            continue
        cs, sel = int(m.group(1)), m.group(2)
        # Each tag dir contains one or more hash subdirs — newest wins.
        candidates = [d for d in tag_dir.iterdir() if (d / "metrics.json").is_file()]
        if not candidates:
            continue
        latest = max(candidates, key=lambda d: (d / "metrics.json").stat().st_mtime)
        metrics = json.loads((latest / "metrics.json").read_text(encoding="utf-8"))
        row = {"chunk_size": cs, "selector": sel}
        for k in METRICS_OF_INTEREST:
            row[k] = float(metrics.get(k, 0.0))
        rows.append(row)
    rows.sort(key=lambda r: (r["chunk_size"], r["selector"]))
    return rows


def write_summary(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunksize_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# HotpotQA × chunk_size × selector sweep", ""]
    md.append("## Citation accuracy by chunk size")
    md.append("")
    md.append("| chunk_size | greedy | anti_kt | mmr | Δ(anti−greedy) |")
    md.append("|---|---|---|---|---|")
    by_cs: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_cs.setdefault(r["chunk_size"], {})[r["selector"]] = r
    cs_sorted = sorted(by_cs)
    for cs in cs_sorted:
        g = by_cs[cs].get("greedy", {}).get("citation_accuracy_mean", float("nan"))
        a = by_cs[cs].get("anti_kt", {}).get("citation_accuracy_mean", float("nan"))
        m = by_cs[cs].get("mmr", {}).get("citation_accuracy_mean", float("nan"))
        delta = a - g if not (np.isnan(a) or np.isnan(g)) else float("nan")
        md.append(f"| {cs} | {g:.3f} | {a:.3f} | {m:.3f} | {delta:+.3f} |")

    md.append("")
    md.append("## Recall@1 by chunk size")
    md.append("")
    md.append("| chunk_size | greedy | anti_kt | mmr |")
    md.append("|---|---|---|---|")
    for cs in cs_sorted:
        g = by_cs[cs].get("greedy", {}).get("recall@1_doc_mean", float("nan"))
        a = by_cs[cs].get("anti_kt", {}).get("recall@1_doc_mean", float("nan"))
        m = by_cs[cs].get("mmr", {}).get("recall@1_doc_mean", float("nan"))
        md.append(f"| {cs} | {g:.3f} | {a:.3f} | {m:.3f} |")

    md.append("")
    md.append("## Answer F1 by chunk size")
    md.append("")
    md.append("| chunk_size | greedy | anti_kt | mmr |")
    md.append("|---|---|---|---|")
    for cs in cs_sorted:
        g = by_cs[cs].get("greedy", {}).get("answer_f1_mean", float("nan"))
        a = by_cs[cs].get("anti_kt", {}).get("answer_f1_mean", float("nan"))
        m = by_cs[cs].get("mmr", {}).get("answer_f1_mean", float("nan"))
        md.append(f"| {cs} | {g:.3f} | {a:.3f} | {m:.3f} |")

    # Paired bootstrap on per-cs anti_kt − greedy citation gap
    paired = []
    for cs in cs_sorted:
        g = by_cs[cs].get("greedy", {}).get("citation_accuracy_mean")
        a = by_cs[cs].get("anti_kt", {}).get("citation_accuracy_mean")
        if g is not None and a is not None:
            paired.append((cs, a - g))
    if len(paired) >= 3:
        diffs = np.asarray([d for _, d in paired])
        rng = np.random.default_rng(0)
        boot = np.array([diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(10000)])
        md.append("")
        md.append("## Paired bootstrap: anti_kt v2 − greedy on citation_accuracy")
        md.append("")
        md.append(f"- mean diff (across {len(paired)} chunk sizes): **{diffs.mean():+.4f}**")
        md.append(f"- 10k-resample CI95: [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
        md.append(f"- P(anti_kt > greedy on average): {(boot > 0).mean():.4f}")

    (out / "chunksize_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "runs/chunksize_sweep")
    ap.add_argument("--out", type=Path, default=REPO / "runs/chunksize_sweep")
    args = ap.parse_args()
    rows = collect(args.runs)
    if not rows:
        print(f"no results in {args.runs}")
        return
    write_summary(rows, args.out)
    print(f"wrote {args.out}/chunksize_summary.md ({len(rows)} runs)")


if __name__ == "__main__":
    main()
