"""One-shot minimal benchmark driver."""

from __future__ import annotations

from pathlib import Path

from astro_cs_rag.config.schema import (
    BenchmarkConfig,
    DetectConfig,
    EvaluateConfig,
    GenerateConfig,
    IndexConfig,
    PathsConfig,
    RerankConfig,
    RetrieveConfig,
    SelectConfig,
)
from astro_cs_rag.pipeline.detect_run import detect_run
from astro_cs_rag.pipeline.evaluate_run import evaluate_run
from astro_cs_rag.pipeline.generate_run import generate_run
from astro_cs_rag.pipeline.index_build import build_index_bundle
from astro_cs_rag.pipeline.rerank_run import rerank_run
from astro_cs_rag.pipeline.retrieve_run import retrieve_run
from astro_cs_rag.pipeline.select_run import select_run


def benchmark_run(cfg: BenchmarkConfig) -> Path:
    index_dir = cfg.index_dir or (cfg.paths.output_dir / "index_bundle")
    idx = IndexConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        paths=PathsConfig(
            corpus_path=cfg.paths.corpus_path,
            output_dir=index_dir,
        ),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retriever=cfg.retriever,
        embedding=cfg.embedding,
        side_indices=cfg.side_indices,
    )
    build_index_bundle(idx)

    retr = RetrieveConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        paths=PathsConfig(
            corpus_path=cfg.paths.corpus_path,
            queries_path=cfg.paths.queries_path,
            output_dir=cfg.paths.output_dir,
        ),
        index_dir=index_dir,
        retriever=cfg.retriever,
        embedding=cfg.embedding,
    )
    run_dir = retrieve_run(retr)

    if cfg.reranker.enabled:
        rerank_run(
            RerankConfig(
                dataset=cfg.dataset,
                seed=cfg.seed,
                run_dir=run_dir,
                index_dir=index_dir,
                reranker=cfg.reranker,
            )
        )

    det = DetectConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        run_dir=run_dir,
        index_dir=index_dir,
        detector=cfg.detector,
    )
    detect_run(det)

    sel = SelectConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        run_dir=run_dir,
        index_dir=index_dir,
        queries_path=cfg.paths.queries_path,
        selector=cfg.selector,
    )
    select_run(sel)

    if cfg.generator.enabled:
        generate_run(
            GenerateConfig(
                dataset=cfg.dataset,
                seed=cfg.seed,
                run_dir=run_dir,
                index_dir=index_dir,
                queries_path=cfg.paths.queries_path,
                generator=cfg.generator,
            )
        )

    ev = EvaluateConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        paths=PathsConfig(
            corpus_path=cfg.paths.corpus_path,
            queries_path=cfg.paths.queries_path,
            gold_path=cfg.paths.gold_path,
            output_dir=cfg.paths.output_dir,
        ),
        run_dir=run_dir,
        index_dir=index_dir,
        metrics=cfg.metrics,
    )
    evaluate_run(ev)
    return run_dir
