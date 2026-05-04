"""Run benchmark variants from merged YAML overrides."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from astro_cs_rag.config.schema import AblationConfig, BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run
from astro_cs_rag.util.dict_merge import deep_merge


def ablation_run(cfg: AblationConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(cfg.base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        msg = "base config must be a YAML mapping"
        raise ValueError(msg)

    summary_rows: list[dict[str, object]] = []
    report_lines = [
        "# Ablation report",
        "",
        "| Variant | Key metrics | Run dir |",
        "|---------|-------------|---------|",
    ]

    for var in cfg.variants:
        merged = deep_merge(base, var.overrides)
        paths = dict(merged.get("paths") or {})
        paths["output_dir"] = str(cfg.output_dir / var.name / "workspace")
        merged["paths"] = paths
        bc = BenchmarkConfig.model_validate(merged)
        run_dir = benchmark_run(bc)
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "variant": var.name,
            "run_dir": str(run_dir),
            "metrics": metrics,
        }
        summary_rows.append(row)
        mrr = metrics.get("mean_reciprocal_rank_doc", 0.0)
        r1 = metrics.get("recall@1_doc_mean", 0.0)
        report_lines.append(
            f"| {var.name} | MRR={mrr:.4f}, recall@1={r1:.4f} | `{run_dir}` |"
        )

    results_path = cfg.output_dir / "ablation_results.jsonl"
    results_path.write_text(
        "\n".join(json.dumps(r) for r in summary_rows) + "\n",
        encoding="utf-8",
    )
    report_path = cfg.output_dir / "ablation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return cfg.output_dir
