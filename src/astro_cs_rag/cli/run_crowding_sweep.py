"""CLI: run the synthetic evidence-crowding sweep.

Usage:
    python -m astro_cs_rag.cli.run_crowding_sweep \\
        --grid phase1 --n_queries 50 \\
        --systems atom_dense chunk_dense \\
        --out_dir runs/d04_crowding/phase1

Writes:
    crowding_sweep_results.jsonl  — one row per (system, query, cell)
    phase_diagram_summary.json    — per-system phase fit on n_distractors
    config.json                   — invocation snapshot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from astro_cs_rag.benchmarks.evidence_crowding.generator import build_dataset
from astro_cs_rag.benchmarks.evidence_crowding.runner import run_cell
from astro_cs_rag.benchmarks.evidence_crowding.sweeps import (
    custom_grid,
    phase1_grid,
    smoke_grid,
)
from astro_cs_rag.diagnostics.phase_transition import fit
from astro_cs_rag.indexing.embedders import HashEmbedder, TrigramEmbedder

_GRIDS = {"phase1": phase1_grid, "smoke": smoke_grid}


def _build_embedder(name: str, model: str):
    if name == "hash":
        return HashEmbedder()
    if name == "trigram":
        return TrigramEmbedder()
    if name == "sbert":
        from astro_cs_rag.indexing.embedders import SentenceEmbedder

        return SentenceEmbedder(model)
    raise ValueError(f"unknown embedder: {name}")


def _summarise(rows: list[dict]) -> dict:
    """Build phase_diagram_summary.

    Slices rows by (system_name, semantic_similarity, token_budget) and
    fits P(success) vs n_distractors_per_gold within each slice.
    """
    bins: dict[tuple[str, str, int], dict[int, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["system_name"], r["semantic_similarity"], r["token_budget"])
        bins[key][r["n_distractors_per_gold"]].append(bool(r["answer_oracle_success"]))
    out: list[dict] = []
    for (sys_name, sim, tb), per_nd in sorted(bins.items()):
        nds = sorted(per_nd)
        rates = [sum(per_nd[nd]) / max(1, len(per_nd[nd])) for nd in nds]
        fitted = fit([float(x) for x in nds], rates)
        out.append(
            {
                "system_name": sys_name,
                "semantic_similarity": sim,
                "token_budget": tb,
                "axis": "n_distractors_per_gold",
                "x": nds,
                "p_success": rates,
                "threshold_C_star": fitted.threshold,
                "auc": fitted.auc,
                "slope_at_threshold": fitted.slope_at_threshold,
                "n_points": fitted.n_points,
            }
        )
    return {"slices": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=[*sorted(_GRIDS), "custom"], default="smoke")
    ap.add_argument("--n_distractors", type=int, nargs="+", default=None)
    ap.add_argument(
        "--similarities", nargs="+", default=None, choices=["low", "medium", "high"]
    )
    ap.add_argument("--token_budgets", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="custom")
    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument(
        "--systems",
        nargs="+",
        default=["atom_dense", "chunk_dense", "atom_iter2"],
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--embedder", choices=["hash", "trigram", "sbert"], default="trigram")
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    embedder = _build_embedder(args.embedder, args.sbert_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "crowding_sweep_results.jsonl"
    summary_path = args.out_dir / "phase_diagram_summary.json"
    config_path = args.out_dir / "config.json"
    rank_dist_path = args.out_dir / "rank_distribution.jsonl"

    config_path.write_text(
        json.dumps(
            {
                "grid": args.grid,
                "n_queries": args.n_queries,
                "systems": args.systems,
                "seed_base": args.seed_base,
                "embedder": args.embedder,
                "sbert_model": args.sbert_model if args.embedder == "sbert" else None,
            },
            indent=2,
        )
    )

    if args.grid == "custom":
        if not (args.n_distractors and args.similarities and args.token_budgets):
            ap.error("--grid custom requires --n_distractors, --similarities, --token_budgets")
        cells_iter = custom_grid(
            n_distractors=args.n_distractors,
            similarities=args.similarities,
            token_budgets=args.token_budgets,
            seed_base=args.seed_base,
            tag=args.tag,
        )
    else:
        cells_iter = _GRIDS[args.grid](seed_base=args.seed_base)
    n_cells = 0
    n_rows = 0
    rows: list[dict] = []
    rank_dists: list[dict] = []
    with open(rows_path, "w", encoding="utf-8") as fh, open(
        rank_dist_path, "w", encoding="utf-8"
    ) as rfh:
        for cell in cells_iter:
            dataset = build_dataset(cell, n_queries=args.n_queries)
            rank_buf: list[dict] = []
            cell_rows = run_cell(
                dataset,
                systems=args.systems,
                embedder=embedder,
                rank_distribution_out=rank_buf,
            )
            for r in cell_rows:
                d = r.model_dump()
                fh.write(json.dumps(d) + "\n")
                rows.append(d)
            for rd in rank_buf:
                rfh.write(json.dumps(rd) + "\n")
                rank_dists.append(rd)
            n_cells += 1
            n_rows += len(cell_rows)

    summary = _summarise(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[crowding] grid={args.grid} cells={n_cells} rows={n_rows}")
    print(f"[crowding] results: {rows_path}")
    print(f"[crowding] summary: {summary_path}")


if __name__ == "__main__":
    main()
