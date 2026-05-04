"""Each retriever mode runs end-to-end through benchmark_run on tiny data."""

from __future__ import annotations

from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run


def _bench_yaml(repo: Path, tmp_path: Path, mode: str, side_indices: dict) -> Path:
    cfg = {
        "dataset": f"tiny_{mode}",
        "seed": 0,
        "paths": {
            "corpus_path": str(repo / "data/tiny/corpus.jsonl"),
            "queries_path": str(repo / "data/tiny/queries.jsonl"),
            "gold_path": str(repo / "data/tiny/gold.jsonl"),
            "output_dir": str(tmp_path / f"out_{mode}"),
        },
        "chunk_size": 120,
        "chunk_overlap": 20,
        "embedding": {"use_hash_embedder": True},
        "side_indices": side_indices,
        "retriever": {"candidate_top_n": 5, "mode": mode},
        "reranker": {"enabled": False},
        "detector": {"window": 3},
        "selector": {"token_budget": 256},
        "generator": {"enabled": True, "provider": "stub"},
        "metrics": {"ks": [1, 3]},
    }
    p = tmp_path / f"bench_{mode}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_bm25_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(_bench_yaml(repo, tmp_path, "bm25", {}), BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "metrics.json").is_file()


def test_dense_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(_bench_yaml(repo, tmp_path, "dense", {}), BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "candidates.jsonl").is_file()


def test_hierarchical_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = _bench_yaml(
        repo, tmp_path, "hierarchical", {"hierarchical": True, "hierarchical_branching": 2}
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "candidates.jsonl").is_file()


def test_splade_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = _bench_yaml(
        repo, tmp_path, "splade", {"splade": True, "splade_use_hash": True}
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "candidates.jsonl").is_file()


def test_late_interaction_mode(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = _bench_yaml(
        repo,
        tmp_path,
        "late_interaction",
        {"late_interaction": True, "late_interaction_use_hash": True},
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    assert (run_dir / "candidates.jsonl").is_file()
