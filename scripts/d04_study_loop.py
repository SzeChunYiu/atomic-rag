"""Self-driving D04 evidence-crowding study.

Each iteration:
  1. Run a crowding sweep on the current n_distractors grid.
  2. Inspect P(success) vs n_distractors per (system, similarity, budget).
  3. For each slice with a *bracketed* C* (some y >= 0.5 and some y < 0.5),
     refine the grid: replace the bracket [x_lo, x_hi] (where the curve
     crosses) with three new points evenly spaced inside it.
  4. Stop when every slice has |x_hi - x_lo| <= --tol or max_iter reached.

Each iteration writes:
  iter_<i>/crowding_sweep_results.jsonl
  iter_<i>/phase_diagram_summary.json
  iter_<i>/REPORT.md

The driver also writes STUDY_LOG.md describing each iteration's
hypothesis, findings, and next-action — so the study explains itself
without me re-reading JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


def _run_sweep(out_dir: Path, n_distractors: list[int], sims: list[str],
               budgets: list[int], n_queries: int, embedder: str, tag: str) -> None:
    cmd = [
        sys.executable, "-m", "astro_cs_rag.cli.run_crowding_sweep",
        "--grid", "custom",
        "--n_distractors", *map(str, n_distractors),
        "--similarities", *sims,
        "--token_budgets", *map(str, budgets),
        "--n_queries", str(n_queries),
        "--systems", "atom_dense", "chunk_dense", "atom_iter2",
        "--embedder", embedder,
        "--tag", tag,
        "--out_dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def _bracket(xs: list[int], ys: list[float]) -> tuple[int, int] | None:
    """Find the [x_lo, x_hi] pair where the curve crosses 0.5 downward."""
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= 0.5 and y1 < 0.5:
            return int(x0), int(x1)
    return None


def _refine_points(lo: int, hi: int) -> list[int]:
    """Three new points strictly inside (lo, hi)."""
    if hi - lo <= 1:
        return []
    step = max(1, (hi - lo) // 4)
    pts = sorted({lo + step, (lo + hi) // 2, hi - step} - {lo, hi})
    return [p for p in pts if lo < p < hi]


def _next_grid(summary: dict, current: list[int], tol: int) -> tuple[list[int], list[dict]]:
    """Decide the next n_distractors grid + per-slice diagnostics."""
    diagnostics = []
    new_pts: set[int] = set(current)
    for s in summary["slices"]:
        bracket = _bracket(s["x"], s["p_success"])
        diag = {
            "system": s["system_name"],
            "similarity": s["semantic_similarity"],
            "budget": s["token_budget"],
            "C_star": s["threshold_C_star"],
            "auc": s["auc"],
            "bracket": bracket,
            "bracket_width": (bracket[1] - bracket[0]) if bracket else None,
        }
        diagnostics.append(diag)
        if bracket and (bracket[1] - bracket[0]) > tol:
            for p in _refine_points(*bracket):
                new_pts.add(p)
    return sorted(new_pts), diagnostics


def _hypothesis(prev_grid: list[int], next_grid: list[int],
                diagnostics: list[dict], tol: int) -> str:
    open_brackets = [d for d in diagnostics if d["bracket"] and d["bracket_width"] > tol]
    if not open_brackets:
        return ("All slices either have C* localised within tol, never crossed 0.5, "
                "or stayed below 0.5 throughout. **Stop.**")
    refined = sorted(set(next_grid) - set(prev_grid))
    return (
        f"{len(open_brackets)} slice(s) still have wide C* brackets. "
        f"Adding n_distractors points {refined} to refine the crossover."
    )


def _log_iteration(log_lines: list[str], it: int, grid: list[int],
                   diagnostics: list[dict], hypothesis: str) -> None:
    log_lines.append(f"## Iteration {it}")
    log_lines.append(f"- grid: `n_distractors = {grid}`")
    log_lines.append("- per-slice C* and brackets:")
    for d in diagnostics:
        cs = "—" if d["C_star"] is None else f"{d['C_star']:.2f}"
        br = "—" if d["bracket"] is None else f"[{d['bracket'][0]}, {d['bracket'][1]}]"
        log_lines.append(
            f"  - {d['system']} | sim={d['similarity']} | tb={d['budget']}: "
            f"C*={cs}, AUC={d['auc']:.3f}, bracket={br}"
        )
    log_lines.append(f"- next-action: {hypothesis}")
    log_lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_dir", type=Path, required=True)
    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument("--init_n_distractors", type=int, nargs="+",
                    default=[0, 2, 5, 10, 20, 50])
    ap.add_argument("--similarities", nargs="+", default=["medium"])
    ap.add_argument("--token_budgets", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--embedder", choices=["hash", "trigram", "sbert"], default="trigram")
    ap.add_argument("--max_iter", type=int, default=3)
    ap.add_argument("--tol", type=int, default=2,
                    help="Stop when every bracket has width <= tol")
    args = ap.parse_args()

    args.study_dir.mkdir(parents=True, exist_ok=True)
    grid = sorted(set(args.init_n_distractors))
    log_lines: list[str] = ["# D04 Self-Driving Study Log", ""]
    final_summary_path: Path | None = None

    for it in range(args.max_iter):
        it_dir = args.study_dir / f"iter_{it}"
        _run_sweep(
            out_dir=it_dir,
            n_distractors=grid,
            sims=args.similarities,
            budgets=args.token_budgets,
            n_queries=args.n_queries,
            embedder=args.embedder,
            tag=f"iter{it}",
        )
        subprocess.run(
            [sys.executable, "scripts/analyze_crowding_sweep.py",
             "--sweep_dir", str(it_dir),
             "--out", str(it_dir / "REPORT.md")],
            check=True,
        )
        summary = json.loads((it_dir / "phase_diagram_summary.json").read_text())
        final_summary_path = it_dir / "phase_diagram_summary.json"
        next_grid, diagnostics = _next_grid(summary, grid, args.tol)
        hypothesis = _hypothesis(grid, next_grid, diagnostics, args.tol)
        _log_iteration(log_lines, it, grid, diagnostics, hypothesis)
        if next_grid == grid:
            log_lines.append("**Converged — no new points to add.**")
            break
        grid = next_grid

    log_lines.append("")
    log_lines.append(f"Final iteration summary: `{final_summary_path}`")
    (args.study_dir / "STUDY_LOG.md").write_text("\n".join(log_lines))
    print(f"[study-loop] log: {args.study_dir/'STUDY_LOG.md'}")


if __name__ == "__main__":
    main()
