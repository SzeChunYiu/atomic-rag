from __future__ import annotations

import json
from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig, ErrorAnalysisConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run
from astro_cs_rag.pipeline.error_analysis_run import error_analysis_run


def test_error_analysis_writes_three_artifacts(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    bench_cfg = tmp_path / "bench.yaml"
    bench_cfg.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_ea",
                "seed": 0,
                "paths": {
                    "corpus_path": str(repo / "data/tiny/corpus.jsonl"),
                    "queries_path": str(repo / "data/tiny/queries.jsonl"),
                    "gold_path": str(repo / "data/tiny/gold.jsonl"),
                    "output_dir": str(tmp_path / "out"),
                },
                "chunk_size": 120,
                "chunk_overlap": 20,
                "embedding": {"use_hash_embedder": True},
                "retriever": {"candidate_top_n": 15},
                "detector": {"window": 5},
                "selector": {"token_budget": 256},
                "metrics": {"ks": [1, 5]},
            }
        ),
        encoding="utf-8",
    )

    run_dir = benchmark_run(load_yaml(bench_cfg, BenchmarkConfig))
    cfg_path = tmp_path / "ea.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_ea",
                "run_dir": str(run_dir),
                "index_dir": str(tmp_path / "out" / "index_bundle"),
                "paths": {
                    "queries_path": str(repo / "data/tiny/queries.jsonl"),
                    "gold_path": str(repo / "data/tiny/gold.jsonl"),
                },
                "baseline_run_dir": None,
            }
        ),
        encoding="utf-8",
    )
    error_analysis_run(load_yaml(cfg_path, ErrorAnalysisConfig))
    assert (run_dir / "error_cases.jsonl").is_file()
    assert (run_dir / "failure_clusters.md").is_file()
    assert (run_dir / "root_cause_report.md").is_file()
    lines = (run_dir / "error_cases.jsonl").read_text(encoding="utf-8").strip().splitlines()
    if lines:
        json.loads(lines[0])
