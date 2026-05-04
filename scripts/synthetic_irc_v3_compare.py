"""Compare anti-kT variants v1 (n_jets=1, merge-order), v2 (n_jets=-1,
SNR-sort), v3 (n_jets=-2, greedy+partner-pullin) against greedy on the
synthetic IRC stress corpus, with paired bootstrap on the mean coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from astro_cs_rag.config.loader import load_yaml  # noqa: E402
from astro_cs_rag.config.schema import BenchmarkConfig  # noqa: E402
from astro_cs_rag.pipeline.benchmark import benchmark_run  # noqa: E402
from synthetic_irc_iter2 import build_corpus  # noqa: E402
from synthetic_irc_experiment import base_cfg, gold_pair_coverage  # noqa: E402


VARIANTS = {
    "greedy": {"mode": "greedy"},
    "anti_kt_v1": {"mode": "anti_kt", "anti_kt_n_jets": 1},
    "anti_kt_v2": {"mode": "anti_kt", "anti_kt_n_jets": -1},
    "anti_kt_v3": {"mode": "anti_kt", "anti_kt_n_jets": -2},
}


def run_one(out_root: Path, paths: dict[str, Path], cs: int, variant: str, seed: int) -> dict:
    cfg_dict = base_cfg(out_root / f"seed{seed}_cs{cs}_{variant}", paths,
                        chunk_size=cs, mode=VARIANTS[variant]["mode"])
    if "anti_kt_n_jets" in VARIANTS[variant]:
        cfg_dict["selector"]["anti_kt_n_jets"] = VARIANTS[variant]["anti_kt_n_jets"]
    cfg_dict["seed"] = seed
    run_out = out_root / f"seed{seed}_cs{cs}_{variant}"
    run_out.mkdir(parents=True, exist_ok=True)
    tmp = run_out / "tmp"
    tmp.mkdir(exist_ok=True)
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    run_dir = benchmark_run(load_yaml(cfg_path, BenchmarkConfig))
    cov = gold_pair_coverage(run_dir, paths["gold"], paths["queries"])
    return {"seed": seed, "chunk_size": cs, "variant": variant, "gold_pair_coverage": cov}


def paired_bootstrap_mean(rows: list[dict], a: str, b: str, *, n_resamples: int = 10000, seed: int = 0) -> dict:
    """Per-(seed,cs) paired comparison; resample (seed,cs) pairs."""
    keyed = {(r["seed"], r["chunk_size"], r["variant"]): r["gold_pair_coverage"] for r in rows}
    pairs = []
    seeds = sorted({r["seed"] for r in rows})
    chunk_sizes = sorted({r["chunk_size"] for r in rows})
    for s in seeds:
        for cs in chunk_sizes:
            va = keyed.get((s, cs, a))
            vb = keyed.get((s, cs, b))
            if va is not None and vb is not None:
                pairs.append(va - vb)
    diffs = np.asarray(pairs)
    if not diffs.size:
        return {}
    rng = np.random.default_rng(seed)
    boot = np.array([diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_resamples)])
    return {
        "a": a, "b": b, "n_pairs": int(len(diffs)),
        "mean_diff": float(diffs.mean()),
        "p_a_higher": float((boot > 0).mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/synthetic_irc_v3_compare")
    ap.add_argument("--n-topics", type=int, default=120)
    ap.add_argument("--n-distractors", type=int, default=240)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--chunk-min", type=int, default=50)
    ap.add_argument("--chunk-max", type=int, default=300)
    ap.add_argument("--chunk-steps", type=int, default=12)
    args = ap.parse_args()

    chunk_sizes = sorted(set(int(x) for x in np.linspace(args.chunk_min, args.chunk_max, args.chunk_steps).round().tolist()))
    rows: list[dict] = []
    for seed in range(args.n_seeds):
        paths = build_corpus(REPO / f"data/synthetic_irc_v3_compare/seed_{seed}",
                             n_topics=args.n_topics, n_distractors=args.n_distractors, seed=seed)
        for cs in chunk_sizes:
            for variant in VARIANTS:
                row = run_one(args.out, paths, cs, variant, seed)
                rows.append(row)
                print(f"seed={seed} cs={cs} {variant}={row['gold_pair_coverage']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"n_seeds": args.n_seeds, "n_chunk_sizes": len(chunk_sizes),
               "n_topics": args.n_topics, "n_distractors": args.n_distractors,
               "comparisons": {}}
    for variant in ("anti_kt_v1", "anti_kt_v2", "anti_kt_v3"):
        summary["comparisons"][f"{variant}_vs_greedy"] = paired_bootstrap_mean(rows, variant, "greedy")
    for v in ("anti_kt_v1", "anti_kt_v2"):
        summary["comparisons"][f"anti_kt_v3_vs_{v}"] = paired_bootstrap_mean(rows, "anti_kt_v3", v)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
