"""Hash embedder end-to-end benchmark (no transformer download)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run


def test_benchmark_tiny_corpus(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_test",
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
                "retriever": {"candidate_top_n": 10},
                "detector": {"window": 5},
                "selector": {"token_budget": 256},
                "metrics": {"ks": [1, 5]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "evidence_atoms.jsonl").is_file()
    assert (run_dir / "selected_context.jsonl").is_file()
    assert (run_dir / "coverage_trace.jsonl").is_file()
    assert (run_dir / "timing.json").is_file()
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    assert "retrieve_seconds" in timing
    assert "evaluate_seconds" in timing
