"""Paired bootstrap test on the synthetic IRC results.

Reads runs/synthetic_irc/synthetic_irc.csv and tests whether anti_kt's
gold_pair_coverage is significantly more chunk-size-robust than greedy / mmr.

Robustness statistic per selector: stdev across chunk_size.
Bootstrap distribution: resample chunk_sizes with replacement (n=1000), recompute
the std for each selector. Compare distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("runs/synthetic_irc/synthetic_irc.csv"))
    ap.add_argument("--out", type=Path, default=Path("runs/synthetic_irc/bootstrap.json"))
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="gold_pair_coverage")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    selectors = sorted({r["selector"] for r in rows})
    by_sel: dict[str, list[tuple[float, float]]] = {s: [] for s in selectors}
    chunk_sizes_seen: set[int] = set()
    for r in rows:
        cs = int(r["chunk_size"])
        chunk_sizes_seen.add(cs)
        by_sel[r["selector"]].append((cs, float(r[args.metric])))

    chunk_sizes = sorted(chunk_sizes_seen)
    rng = np.random.default_rng(args.seed)

    boot_stats: dict[str, list[float]] = {s: [] for s in selectors}
    for _ in range(args.n_resamples):
        sample_idx = rng.integers(0, len(chunk_sizes), size=len(chunk_sizes))
        sampled_cs = [chunk_sizes[i] for i in sample_idx]
        for sel in selectors:
            vals = [v for cs, v in by_sel[sel] if cs in sampled_cs]
            # Take only one occurrence per chunk_size (already aggregated)
            seen: dict[int, float] = {}
            for cs, v in by_sel[sel]:
                if cs in sampled_cs and cs not in seen:
                    seen[cs] = v
            chosen = [seen[cs] for cs in sampled_cs if cs in seen]
            if len(chosen) >= 2:
                boot_stats[sel].append(float(statistics.pstdev(chosen)))

    summary: dict[str, object] = {"metric": args.metric, "n_resamples": args.n_resamples, "selectors": {}}
    for sel, vals in boot_stats.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        summary["selectors"][sel] = {  # type: ignore[index]
            "stdev_mean": float(arr.mean()),
            "stdev_p05": float(np.percentile(arr, 5)),
            "stdev_p95": float(np.percentile(arr, 95)),
        }
    if "anti_kt" in boot_stats and "greedy" in boot_stats:
        a = np.asarray(boot_stats["anti_kt"])
        g = np.asarray(boot_stats["greedy"])
        n = min(len(a), len(g))
        if n > 0:
            diff = a[:n] - g[:n]
            summary["paired_diff_anti_minus_greedy"] = {
                "mean": float(diff.mean()),
                "p_anti_lower_than_greedy": float((diff < 0).mean()),
            }

    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
