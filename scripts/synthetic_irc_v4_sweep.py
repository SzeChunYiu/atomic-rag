"""Synthetic IRC sweep for v4 (score-gated partner pull-in).

Compares: greedy, v3 (alpha=0), v4 at alpha ∈ {0.3, 0.5, 0.7, 0.9} (with median floor).
Re-uses corpus + paired-bootstrap from synthetic_irc_v3_compare.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from synthetic_irc_v3_compare import paired_bootstrap_mean  # noqa: E402


VARIANTS = {
    "greedy": {"mode": "greedy"},
    "anti_kt_v3": {"mode": "anti_kt", "anti_kt_n_jets": -2, "alpha": 0.0, "median": False},
    "anti_kt_v4_a03": {"mode": "anti_kt", "anti_kt_n_jets": -2, "alpha": 0.3, "median": True},
    "anti_kt_v4_a05": {"mode": "anti_kt", "anti_kt_n_jets": -2, "alpha": 0.5, "median": True},
    "anti_kt_v4_a07": {"mode": "anti_kt", "anti_kt_n_jets": -2, "alpha": 0.7, "median": True},
    "anti_kt_v4_a09": {"mode": "anti_kt", "anti_kt_n_jets": -2, "alpha": 0.9, "median": True},
}


def run_one(out_root: Path, paths: dict[str, Path], cs: int, variant: str, seed: int) -> dict:
    spec = VARIANTS[variant]
    cfg_dict = base_cfg(out_root / f"seed{seed}_cs{cs}_{variant}", paths,
                        chunk_size=cs, mode=spec["mode"])
    if spec.get("anti_kt_n_jets") is not None:
        cfg_dict["selector"]["anti_kt_n_jets"] = spec["anti_kt_n_jets"]
        cfg_dict["selector"]["partner_score_gate_alpha"] = float(spec.get("alpha", 0.0))
        cfg_dict["selector"]["partner_use_median_floor"] = bool(spec.get("median", False))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/synthetic_irc_v4_sweep")
    ap.add_argument("--n-topics", type=int, default=120)
    ap.add_argument("--n-distractors", type=int, default=240)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--chunk-min", type=int, default=50)
    ap.add_argument("--chunk-max", type=int, default=300)
    ap.add_argument("--chunk-steps", type=int, default=8)
    args = ap.parse_args()

    chunk_sizes = sorted(set(int(x) for x in np.linspace(args.chunk_min, args.chunk_max, args.chunk_steps).round().tolist()))
    rows: list[dict] = []
    for seed in range(args.n_seeds):
        paths = build_corpus(REPO / f"data/synthetic_irc_v4_sweep/seed_{seed}",
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
    for variant in ("anti_kt_v3", "anti_kt_v4_a03", "anti_kt_v4_a05", "anti_kt_v4_a07", "anti_kt_v4_a09"):
        summary["comparisons"][f"{variant}_vs_greedy"] = paired_bootstrap_mean(rows, variant, "greedy")
    for v in ("anti_kt_v4_a03", "anti_kt_v4_a05", "anti_kt_v4_a07", "anti_kt_v4_a09"):
        summary["comparisons"][f"{v}_vs_v3"] = paired_bootstrap_mean(rows, v, "anti_kt_v3")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
