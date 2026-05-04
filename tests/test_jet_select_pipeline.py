"""End-to-end pipeline with anti-kT selector mode."""

from __future__ import annotations

from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run


def test_anti_kt_selector_runs_end_to_end(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_anti_kt",
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
                "retriever": {"candidate_top_n": 5, "mode": "fusion_rrf"},
                "reranker": {"enabled": False},
                "detector": {"window": 3},
                "selector": {
                    "token_budget": 256,
                    "mode": "anti_kt",
                    "anti_kt_R": 1.0,
                    "anti_kt_n_jets": 1,
                },
                "generator": {"enabled": True, "provider": "stub"},
                "metrics": {"ks": [1, 3]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "selected_context.jsonl").is_file()
    sel_lines = (run_dir / "selected_context.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"anti_kt' in line for line in sel_lines)


def test_mmr_selector_runs_end_to_end(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_mmr",
                "seed": 0,
                "paths": {
                    "corpus_path": str(repo / "data/tiny/corpus.jsonl"),
                    "queries_path": str(repo / "data/tiny/queries.jsonl"),
                    "gold_path": str(repo / "data/tiny/gold.jsonl"),
                    "output_dir": str(tmp_path / "out_mmr"),
                },
                "chunk_size": 120,
                "chunk_overlap": 20,
                "embedding": {"use_hash_embedder": True},
                "retriever": {"candidate_top_n": 5},
                "reranker": {"enabled": False},
                "detector": {"window": 3},
                "selector": {
                    "token_budget": 256,
                    "mode": "mmr",
                    "mmr_lambda": 0.6,
                },
                "generator": {"enabled": True, "provider": "stub"},
                "metrics": {"ks": [1, 3]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "metrics.json").is_file()
