"""CLI/backend driver for error mining."""

from __future__ import annotations

from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.config.schema import ErrorAnalysisConfig
from astro_cs_rag.data.loaders import load_gold_jsonl, load_queries_jsonl, merge_gold_into_queries
from astro_cs_rag.reporting.error_analysis import analyze_run_errors


def error_analysis_run(cfg: ErrorAnalysisConfig) -> Path:
    qp = cfg.paths.queries_path
    if qp is None:
        msg = "queries_path missing"
        raise ValueError(msg)
    queries = load_queries_jsonl(qp)
    if cfg.paths.gold_path is not None:
        gold = load_gold_jsonl(cfg.paths.gold_path)
        queries = merge_gold_into_queries(queries, gold)

    payload = analyze_run_errors(
        run_dir=cfg.run_dir,
        index_dir=cfg.index_dir,
        queries=queries,
        baseline_run_dir=cfg.baseline_run_dir,
    )

    writer = ArtifactWriter.attach(cfg.run_dir)
    writer.write_jsonl("error_cases.jsonl", payload["error_cases"])
    writer.write_markdown("failure_clusters.md", payload["failure_clusters_md"])
    writer.write_markdown("root_cause_report.md", payload["root_cause_report_md"])
    return cfg.run_dir
