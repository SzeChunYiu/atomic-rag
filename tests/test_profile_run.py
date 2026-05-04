"""Profile run wires retrieve output → query_shapes.jsonl + archetype_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run
from astro_cs_rag.pipeline.profile_run import profile_run


def test_profile_writes_archetype_summary(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_profile",
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
                "retriever": {"candidate_top_n": 5},
                "reranker": {"enabled": False},
                "detector": {"window": 3},
                "selector": {"token_budget": 256},
                "generator": {"enabled": True, "provider": "stub"},
                "metrics": {"ks": [1, 3]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    profile_run(run_dir, gold_path=repo / "data/tiny/gold.jsonl", ks=(1, 3))

    qsh = run_dir / "query_shapes.jsonl"
    summary = run_dir / "archetype_summary.json"
    assert qsh.is_file()
    assert summary.is_file()

    rows = [json.loads(line) for line in qsh.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 1
    assert all("archetype" in r for r in rows)

    summary_data = json.loads(summary.read_text(encoding="utf-8"))
    assert "archetypes" in summary_data
    assert summary_data["query_count"] == float(len(rows))
