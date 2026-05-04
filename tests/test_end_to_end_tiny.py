"""End-to-end tiny benchmark with stub generator (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run


def test_full_pipeline_with_stub_generator(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "bench.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "tiny_e2e",
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
                "reranker": {"enabled": False},
                "detector": {"window": 5},
                "selector": {"token_budget": 256},
                "generator": {
                    "enabled": True,
                    "provider": "stub",
                },
                "metrics": {"ks": [1, 3]},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    expected = [
        "config.yaml",
        "manifest.json",
        "candidates.jsonl",
        "evidence_atoms.jsonl",
        "selected_context.jsonl",
        "generated_answers.jsonl",
        "generation_meta.json",
        "metrics.json",
        "report.md",
        "timing.json",
    ]
    for name in expected:
        assert (run_dir / name).is_file(), f"missing {name}"

    answers = (run_dir / "generated_answers.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(answers) >= 1
    parsed = [json.loads(line) for line in answers]
    assert all("[E1]" in row["answer_text"] or row["answer_text"] == "I don't know." for row in parsed)
