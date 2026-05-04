"""Rerank candidates with a cross-encoder; rewrite candidates.jsonl."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.atoms.schemas import Candidate
from astro_cs_rag.config.schema import RerankConfig
from astro_cs_rag.indexing.io import load_chunks_jsonl
from astro_cs_rag.pipeline.retrieve_run import load_candidates_jsonl
from astro_cs_rag.reranking.cross_encoder import (
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
)


def _build_reranker(model_name: str, batch_size: int) -> Reranker:
    if model_name == "identity":
        return IdentityReranker()
    return CrossEncoderReranker(model_name=model_name, batch_size=batch_size)


def rerank_run(cfg: RerankConfig) -> Path:
    if not cfg.reranker.enabled:
        return cfg.run_dir
    candidates = load_candidates_jsonl(cfg.run_dir / "candidates.jsonl")
    chunks = load_chunks_jsonl(cfg.index_dir / "chunks.jsonl")
    text_by_chunk = {c.chunk_id: c.text for c in chunks}

    by_q: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_q[c.query_id].append(c)

    queries_path = cfg.run_dir / "config.yaml"
    if not queries_path.is_file():
        msg = "rerank requires a prior retrieve run with config.yaml"
        raise ValueError(msg)

    queries_text = _query_text_lookup(cfg.run_dir)
    reranker = _build_reranker(cfg.reranker.model_name, cfg.reranker.batch_size)

    writer = ArtifactWriter.attach(cfg.run_dir)
    rerank_seconds = 0.0
    new_candidates: list[Candidate] = []
    for qid, parts in by_q.items():
        in_pool = sorted(parts, key=lambda x: -x.raw_score)[: cfg.reranker.top_n_in]
        passages = [text_by_chunk.get(p.chunk_id, "") for p in in_pool]
        t0 = time.perf_counter()
        scores = reranker.score(queries_text.get(qid, ""), passages)
        rerank_seconds += time.perf_counter() - t0

        scored = list(zip(in_pool, scores.tolist(), strict=True))
        scored.sort(key=lambda x: -x[1])
        out_pool = scored[: cfg.reranker.top_n_out]

        for rank, (cand, s) in enumerate(out_pool, start=1):
            new_candidates.append(
                cand.model_copy(
                    update={
                        "raw_score": float(s),
                        "retriever": f"{cand.retriever}+rerank({reranker.name})",
                        "rank": rank,
                        "metadata": {
                            **cand.metadata,
                            "pre_rerank_score": cand.raw_score,
                            "pre_rerank_rank": cand.rank,
                        },
                    }
                )
            )

    writer.write_jsonl("candidates.jsonl", new_candidates)
    writer.write_json(
        "rerank_meta.json",
        {
            "model_name": cfg.reranker.model_name,
            "top_n_in": cfg.reranker.top_n_in,
            "top_n_out": cfg.reranker.top_n_out,
            "rerank_seconds": rerank_seconds,
        },
    )
    return cfg.run_dir


def _query_text_lookup(run_dir: Path) -> dict[str, str]:
    """Reload original query text from the config snapshot's queries_path.

    Relies on the standard convention that retrieve_run writes config.yaml
    pointing to the queries file. Avoids re-passing it through every stage.
    """
    import yaml

    cfg_raw = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    paths = cfg_raw.get("paths") or {}
    qp = paths.get("queries_path")
    if qp is None:
        return {}
    from astro_cs_rag.data.loaders import load_queries_jsonl

    return {q.query_id: q.text for q in load_queries_jsonl(Path(qp))}
