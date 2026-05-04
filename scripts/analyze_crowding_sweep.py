"""Analyze a crowding sweep and emit a markdown findings report.

Reads `crowding_sweep_results.jsonl` + `phase_diagram_summary.json` from
one or more sweep dirs and writes a single REPORT.md that highlights:
  - C* and AUC per (system, similarity, budget)
  - the smoking-gun "selection fails even at nd=0" detector
  - cross-embedder comparison if more than one sweep dir is supplied

Usage:
    python scripts/analyze_crowding_sweep.py \\
        --sweep_dir runs/d04_crowding/phase1_hash \\
        --sweep_dir runs/d04_crowding/phase1_sbert \\
        --out runs/d04_crowding/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _load_sweep(sweep_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (sweep_dir / "crowding_sweep_results.jsonl").read_text().splitlines()
    ]
    summary = json.loads((sweep_dir / "phase_diagram_summary.json").read_text())
    config = json.loads((sweep_dir / "config.json").read_text())
    rd_path = sweep_dir / "rank_distribution.jsonl"
    rank_dists: list[dict] = []
    if rd_path.exists():
        rank_dists = [json.loads(line) for line in rd_path.read_text().splitlines()]
    return {
        "dir": sweep_dir, "rows": rows, "summary": summary,
        "config": config, "rank_dists": rank_dists,
    }


def _aggregate_rank_dist(rank_dists: list[dict]) -> list[dict]:
    """Median across cells of per-role rank stats."""
    by_role: dict[str, dict[str, list[float]]] = {}
    for rd in rank_dists:
        for s in rd["by_role"]:
            entry = by_role.setdefault(s["role"], {
                "median_global_rank": [],
                "median_within_role_rank": [],
                "pct_in_top1": [],
                "pct_in_top5": [],
                "n": [],
            })
            for k in entry:
                entry[k].append(s[k])
    out = []
    for role in sorted(by_role):
        e = by_role[role]
        out.append(
            {
                "role": role,
                "median_global_rank": sum(e["median_global_rank"]) / len(e["median_global_rank"]),
                "median_within_role_rank": sum(e["median_within_role_rank"])
                / len(e["median_within_role_rank"]),
                "pct_in_top1": sum(e["pct_in_top1"]) / len(e["pct_in_top1"]),
                "pct_in_top5": sum(e["pct_in_top5"]) / len(e["pct_in_top5"]),
                "n_cells": len(e["n"]),
            }
        )
    return out


def _format_rank_table(agg: list[dict]) -> str:
    headers = (
        "| role | median global rank | median within-role rank | %top1 | %top5 |"
    )
    sep = "|---|---|---|---|---|"
    lines = [headers, sep]
    for s in agg:
        lines.append(
            f"| {s['role']} | {s['median_global_rank']:.1f} | "
            f"{s['median_within_role_rank']:.1f} | {s['pct_in_top1']:.2f} | "
            f"{s['pct_in_top5']:.2f} |"
        )
    return "\n".join(lines)


def _baseline_failure_table(rows: list[dict]) -> list[tuple[str, int, float, float]]:
    """For nd=0 (no distractors), aggregate (system, budget) → P(success), gold_recall."""
    bins: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r["n_distractors_per_gold"] == 0:
            bins[(r["system_name"], r["token_budget"])].append(r)
    out = []
    for (sys_name, tb), rs in sorted(bins.items()):
        p = mean(float(r["support_chain_complete"]) for r in rs)
        recall = mean(float(r["gold_atom_recall_at_k"]) for r in rs)
        out.append((sys_name, tb, p, recall))
    return out


def _phase_table(summary: dict) -> list[dict]:
    return summary["slices"]


def _format_phase_table(slices: list[dict]) -> str:
    headers = "| system | similarity | budget | C* | AUC | slope | x | P(success) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [headers, sep]
    for s in slices:
        cs = "—" if s["threshold_C_star"] is None else f"{s['threshold_C_star']:.1f}"
        ps = "[" + ", ".join(f"{p:.2f}" for p in s["p_success"]) + "]"
        xs = "[" + ", ".join(str(x) for x in s["x"]) + "]"
        lines.append(
            f"| {s['system_name']} | {s['semantic_similarity']} | {s['token_budget']} | "
            f"{cs} | {s['auc']:.3f} | {s['slope_at_threshold']:.4f} | {xs} | {ps} |"
        )
    return "\n".join(lines)


def _format_baseline_table(table: list[tuple[str, int, float, float]]) -> str:
    headers = "| system | token_budget | P(support_chain_complete) @ nd=0 | gold_atom_recall@200 |"
    sep = "|---|---|---|---|"
    lines = [headers, sep]
    for sys_name, tb, p, recall in table:
        lines.append(f"| {sys_name} | {tb} | {p:.3f} | {recall:.3f} |")
    return "\n".join(lines)


def _compose_report(sweeps: list[dict]) -> str:
    out = ["# D04 Evidence Crowding — Automated Findings", ""]
    for sw in sweeps:
        cfg = sw["config"]
        tag = f"{cfg.get('embedder', 'hash')}"
        if cfg.get("sbert_model"):
            tag += f" ({cfg['sbert_model']})"
        out.append(f"## Sweep: `{sw['dir']}` — embedder={tag}")
        out.append(
            f"- grid: **{cfg['grid']}**, n_queries={cfg['n_queries']}, "
            f"systems={cfg['systems']}, seed_base={cfg['seed_base']}"
        )
        out.append(f"- rows: {len(sw['rows'])}")
        out.append("")
        out.append("### nd=0 baseline — should be near-perfect")
        baseline = _baseline_failure_table(sw["rows"])
        out.append(_format_baseline_table(baseline))
        flagged = [(s, tb, p, r) for s, tb, p, r in baseline if p < 0.5 and r > 0.9]
        if flagged:
            out.append("")
            out.append("> **smoking-gun**: gold atoms ARE in the candidate pool but selection drops them.")
            out.append("> This isolates the failure to the selection layer (cos-vs-bridge), not retrieval.")
            for s, tb, p, r in flagged:
                out.append(f"> - `{s}` @ tb={tb}: P={p:.2f}, recall={r:.2f}")
        out.append("")
        if sw.get("rank_dists"):
            agg = _aggregate_rank_dist(sw["rank_dists"])
            out.append("### Per-role atom rank (median across cells)")
            out.append(
                "Sanity check: a useful embedder ranks gold atoms near the top. "
                "Median global rank ~N/2 means uniform-random — embedder is broken."
            )
            out.append(_format_rank_table(agg))
            out.append("")
        out.append("### Phase fits per (system, similarity, budget)")
        out.append(_format_phase_table(_phase_table(sw["summary"])))
        out.append("")

    if len(sweeps) > 1:
        out.append("## Cross-embedder AUC comparison (mean over slices)")
        out.append("| sweep | system | mean AUC | mean C* (drop nulls) |")
        out.append("|---|---|---|---|")
        for sw in sweeps:
            tag = sw["config"].get("embedder", "hash")
            by_sys: dict[str, list[dict]] = defaultdict(list)
            for s in sw["summary"]["slices"]:
                by_sys[s["system_name"]].append(s)
            for sys_name, slices in sorted(by_sys.items()):
                aucs = [s["auc"] for s in slices]
                cs = [s["threshold_C_star"] for s in slices if s["threshold_C_star"] is not None]
                cs_str = f"{mean(cs):.2f}" if cs else "—"
                out.append(f"| {tag} | {sys_name} | {mean(aucs):.3f} | {cs_str} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sweeps = [_load_sweep(d) for d in args.sweep_dir]
    args.out.write_text(_compose_report(sweeps))
    print(f"wrote {args.out} ({sum(len(s['rows']) for s in sweeps)} rows analyzed)")


if __name__ == "__main__":
    main()
