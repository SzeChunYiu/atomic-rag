"""Retrieve fused candidates for each query."""

from __future__ import annotations

import time
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.atoms.schemas import Candidate, RunManifest
from astro_cs_rag.config.schema import RetrieveConfig
from astro_cs_rag.data.loaders import load_queries_jsonl
from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
from astro_cs_rag.retrieval.candidates import build_candidates
from astro_cs_rag.retrieval.fusion import rank_by_score, reciprocal_rank_fusion
from astro_cs_rag.retrieval.strategies import (
    bm25_scores,
    dense_scores,
    fusion_rrf_scores,
    hierarchical_scores,
    late_interaction_scores,
    lockin_scores,
    splade_scores,
)


def retrieve_run(cfg: RetrieveConfig) -> Path:
    if cfg.paths.queries_path is None:
        msg = "retrieve requires paths.queries_path"
        raise ValueError(msg)
    chunks, bm25, dense, meta = load_index_bundle(cfg.index_dir)
    embedder = embedder_from_meta(meta, cfg.embedding)
    queries = load_queries_jsonl(cfg.paths.queries_path)

    # Load pre-cached paraphrases (lockin retriever only).
    paraphrase_cache: dict[str, list[str]] = {}
    cache_path = cfg.retriever.lockin_paraphrase_cache_path
    if cfg.retriever.mode == "lockin" and cache_path is not None:
        import json
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            paraphrase_cache[rec["query_id"]] = list(rec["paraphrases"])

    t0 = time.perf_counter()
    writer = ArtifactWriter(cfg.paths.output_dir)
    writer.touch_started()
    writer.write_config_snapshot(cfg)

    rows: list[Candidate] = []
    score_rows: list[dict[str, object]] = []
    mode = cfg.retriever.mode

    # Batch-encode all queries once. BGE-M3 has O(50ms) per-call overhead from
    # CUDA kernel launch + tokenization; encoding 1000 queries one-at-a-time
    # was the dominant 63 s in retrieve_seconds. One large batch is ~10× faster.
    needs_embedding = mode in {"fusion_rrf", "dense", "hierarchical", "lockin"}
    if needs_embedding:
        all_q_embs = embedder.encode([q.text for q in queries])
    else:
        all_q_embs = [None] * len(queries)
    for q, q_emb in zip(queries, all_q_embs, strict=True):
        if mode == "fusion_rrf":
            scores = fusion_rrf_scores(bm25, dense, q.text, q_emb)
            label = "fusion_rrf"
        elif mode == "bm25":
            scores = bm25_scores(bm25, q.text)
            label = "bm25"
        elif mode == "dense":
            scores = dense_scores(dense, q_emb)
            label = "dense"
        elif mode == "late_interaction":
            scores = late_interaction_scores(
                cfg.index_dir,
                q.text,
                use_hash_encoder=cfg.embedding.use_hash_embedder,
                encoder_model_name=cfg.embedding.model_name,
            )
            label = "late_interaction"
        elif mode == "hierarchical":
            scores = hierarchical_scores(cfg.index_dir, q_emb)
            label = "hierarchical"
        elif mode == "splade":
            scores = splade_scores(
                cfg.index_dir,
                q.text,
                use_hash_backend=cfg.embedding.use_hash_embedder,
            )
            label = "splade"
        elif mode == "lockin":
            paras = paraphrase_cache.get(q.query_id)
            scores = lockin_scores(
                bm25,
                dense,
                embedder,
                q.text,
                n_paraphrases=cfg.retriever.lockin_n_paraphrases,
                use_fixed_pattern_phase=cfg.retriever.lockin_use_fixed_pattern_phase,
                paraphrase_settings=None,
                paraphrases=paras,
            )
            label = "lockin"
        else:
            msg = f"unknown retriever mode {mode}"
            raise ValueError(msg)

        part = build_candidates(
            q.query_id,
            scores,
            top_n=cfg.retriever.candidate_top_n,
            retriever_label=label,
        )
        rows.extend(part)
        score_rows.append(
            {
                "query_id": q.query_id,
                "mode": label,
                "top": rank_by_score(scores)[:10],
            }
        )

    writer.write_jsonl("candidates.jsonl", rows)
    writer.write_jsonl("scores.jsonl", score_rows)
    manifest = RunManifest(
        run_id=writer.run_id,
        dataset_name=cfg.dataset,
        corpus_doc_count=len({c.doc_id for c in chunks}),
        query_count=len(queries),
        chunk_count=len(chunks),
        seed=cfg.seed,
        paths={
            "index_dir": str(cfg.index_dir),
            "queries_path": str(cfg.paths.queries_path),
        },
    )
    writer.write_manifest(manifest)
    retrieve_seconds = time.perf_counter() - t0
    writer.write_json("timing.json", {"retrieve_seconds": retrieve_seconds})
    return writer.run_path


def load_candidates_jsonl(path: Path) -> list[Candidate]:
    rows: list[Candidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Candidate.model_validate_json(line))
    return rows


def rankings_from_candidates(
    candidates: list[Candidate],
) -> dict[str, list[str]]:
    """Stable ordering: ascending rank field then chunk_id."""
    by_q: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_q.setdefault(c.query_id, []).append(c)
    out: dict[str, list[str]] = {}
    for qid, parts in by_q.items():
        parts_sorted = sorted(
            parts,
            key=lambda x: (x.rank is None, x.rank or 10**9, x.chunk_id),
        )
        out[qid] = [p.chunk_id for p in parts_sorted]
    return out
