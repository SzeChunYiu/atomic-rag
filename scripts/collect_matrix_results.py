"""Aggregate matrix-run metrics into a single CSV + markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(matrix_root: Path, out_csv: Path, out_md: Path) -> None:
    rows: list[dict] = []
    for run_dir in sorted(matrix_root.iterdir()):
        if not run_dir.is_dir():
            continue
        # Each config produces one or more run subdirectories (one per invocation).
        run_subdirs = [p for p in run_dir.iterdir() if (p / "metrics.json").is_file()]
        if not run_subdirs:
            continue
        latest = max(run_subdirs, key=lambda p: p.stat().st_mtime)
        metrics = json.loads((latest / "metrics.json").read_text(encoding="utf-8"))
        name = run_dir.name
        if "__" in name:
            ds, retr, sel = name.split("__")
        else:
            ds = retr = sel = name
        rows.append(
            {
                "dataset": ds,
                "retriever": retr,
                "selector": sel,
                "run_dir": str(latest),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
        )

    if not rows:
        out_csv.write_text("", encoding="utf-8")
        out_md.write_text("(no matrix runs found)\n", encoding="utf-8")
        return

    keys = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    headline = ["recall@1_doc_mean", "recall@5_doc_mean", "recall@10_doc_mean", "answer_em_mean", "answer_f1_mean", "citation_accuracy_mean"]
    md = ["# Experiment-matrix results", ""]
    md.append("| dataset | retriever | selector | " + " | ".join(headline) + " |")
    md.append("|" + "|".join(["---"] * (3 + len(headline))) + "|")
    for r in rows:
        cells = [r["dataset"], r["retriever"], r["selector"]]
        for h in headline:
            v = r.get(h)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
        md.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-root", type=Path, default=Path("runs/matrix"))
    ap.add_argument("--out-csv", type=Path, default=Path("runs/matrix_results.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("runs/matrix_results.md"))
    args = ap.parse_args()
    collect(args.matrix_root, args.out_csv, args.out_md)
    print(f"wrote {args.out_csv} + {args.out_md}")


if __name__ == "__main__":
    main()
