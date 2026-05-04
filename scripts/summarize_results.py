"""Auto-generate a Markdown results table from all completed runs.

Scans runs-root for metrics.json files, joins with config.yaml for labels,
sorts by F1 descending, writes a Markdown table.

Usage:
    python scripts/summarize_results.py --runs-root runs/phase2 --out results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("runs/analysis/results_summary.md"))
    ap.add_argument("--baseline-dir", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for metrics_path in sorted(args.runs_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.name == "index_bundle":
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        em = m.get("answer_em_mean")
        f1 = m.get("answer_f1_mean")
        cit = m.get("citation_accuracy_mean")
        if em is None or f1 is None:
            continue

        cfg_path = run_dir / "config.yaml"
        label = run_dir.parent.name if cfg_path.is_file() else run_dir.name

        # Read dataset from config
        dataset = "?"
        prompt = "?"
        topk = "?"
        budget = "?"
        if cfg_path.is_file():
            text = cfg_path.read_text(encoding="utf-8")
            if _HAS_YAML:
                try:
                    cfg_dict = _yaml.safe_load(text) or {}
                    raw_ds = str(cfg_dict.get("dataset", "?"))
                    dataset = raw_ds.split("_")[0] if raw_ds != "?" else "?"
                    prompt = str(cfg_dict.get("generator", {}).get("prompt_style", "?"))
                    topk = str(cfg_dict.get("retriever", {}).get("candidate_top_n", "?"))
                    budget = str(cfg_dict.get("selector", {}).get("token_budget", "?"))
                except Exception:
                    pass
            else:
                for line in text.splitlines():
                    if line.strip().startswith("dataset:"):
                        dataset = line.split(":", 1)[1].strip().split("_")[0]
                    if "prompt_style:" in line:
                        prompt = line.split(":", 1)[1].strip()
                    if "candidate_top_n:" in line:
                        topk = line.split(":", 1)[1].strip()
                    if "token_budget:" in line:
                        budget = line.split(":", 1)[1].strip()

        rows.append({
            "label": label,
            "dataset": dataset,
            "prompt": prompt,
            "topk": topk,
            "budget": budget,
            "em": em,
            "f1": f1,
            "cit": cit or 0.0,
        })

    rows.sort(key=lambda r: (-r["f1"], r["dataset"]))

    lines = [
        "# Results Summary",
        "",
        f"Generated from `{args.runs_root}` — {len(rows)} runs",
        "",
        "| Run | Dataset | Prompt | topK | Budget | EM | F1 | Cit |",
        "|-----|---------|--------|------|--------|----|----|-----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['dataset']} | {r['prompt']} | {r['topk']} "
            f"| {r['budget']} | {r['em']:.3f} | {r['f1']:.4f} | {r['cit']:.3f} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(rows)} rows to {args.out}")

    # Print to stdout too
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
