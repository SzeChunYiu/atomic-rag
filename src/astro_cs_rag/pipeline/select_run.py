"""Greedy budget selection over evidence atoms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.config.schema import SelectConfig
from astro_cs_rag.data.loaders import load_queries_jsonl
from astro_cs_rag.indexing.dense import DenseIndex
from astro_cs_rag.indexing.io import load_chunks_jsonl
from astro_cs_rag.pipeline.detect_run import load_evidence_atoms_jsonl
from astro_cs_rag.selection.clean_rag import clean_select
from astro_cs_rag.selection.greedy import greedy_select
from astro_cs_rag.selection.jet_select import jet_select
from astro_cs_rag.selection.mmr import mmr_select


def select_run(cfg: SelectConfig) -> Path:
    atoms = load_evidence_atoms_jsonl(cfg.run_dir / "evidence_atoms.jsonl")
    chunks = load_chunks_jsonl(cfg.index_dir / "chunks.jsonl")
    cmap = {c.chunk_id: c for c in chunks}

    queries_by_id: dict[str, str] = {}
    if cfg.queries_path is not None:
        for q in load_queries_jsonl(cfg.queries_path):
            queries_by_id[q.query_id] = q.text

    embeddings_by_id: dict = {}
    if cfg.selector.mode in {"anti_kt", "mmr", "clean_rag"}:
        try:
            dense = DenseIndex.load(cfg.index_dir)
            for cid, vec in zip(dense.chunk_ids, dense.embeddings, strict=True):
                embeddings_by_id[cid] = vec
        except FileNotFoundError:
            pass

    # Per-query embeddings for CLEAN-RAG: re-encode using BGE-M3 (or whatever
    # embedder the index was built with). Loaded lazily via index meta.
    query_embeddings: dict = {}
    if cfg.selector.mode == "clean_rag" and queries_by_id:
        from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
        from astro_cs_rag.config.schema import EmbeddingSettings
        _, _, _, meta = load_index_bundle(cfg.index_dir)
        embedder = embedder_from_meta(meta, EmbeddingSettings())
        qids = list(queries_by_id.keys())
        embs = embedder.encode([queries_by_id[q] for q in qids])
        for qid, emb in zip(qids, embs, strict=True):
            query_embeddings[qid] = emb

    by_q: dict[str, list] = defaultdict(list)
    for a in atoms:
        by_q[a.query_id].append(a)

    selected_rows: list[object] = []
    dropped_rows: list[dict[str, object]] = []
    trace_all: list[dict[str, object]] = []

    if cfg.selector.mode == "anti_kt":
        sel, dropped, trace = jet_select(
            atoms,
            cmap,
            embeddings_by_id,
            token_budget=cfg.selector.token_budget,
            R=cfg.selector.anti_kt_R,
            n_jets=cfg.selector.anti_kt_n_jets,
            partner_score_gate_alpha=cfg.selector.partner_score_gate_alpha,
            partner_use_median_floor=cfg.selector.partner_use_median_floor,
        )
        selected_rows.extend(sel)
        dropped_rows.extend(dropped)
        trace_all.extend(trace)
    elif cfg.selector.mode == "mmr":
        sel, dropped, trace = mmr_select(
            atoms,
            cmap,
            embeddings_by_id,
            token_budget=cfg.selector.token_budget,
            lambda_param=cfg.selector.mmr_lambda,
        )
        selected_rows.extend(sel)
        dropped_rows.extend(dropped)
        trace_all.extend(trace)
    elif cfg.selector.mode == "clean_rag":
        sel, dropped, trace = clean_select(
            atoms=atoms,
            chunks_by_id=cmap,
            embeddings_by_id=embeddings_by_id,
            query_embeddings=query_embeddings,
            token_budget=cfg.selector.token_budget,
            gain=cfg.selector.clean_rag_gain,
            residual_floor=cfg.selector.clean_rag_residual_floor,
            max_iters=cfg.selector.clean_rag_max_iters,
        )
        selected_rows.extend(sel)
        dropped_rows.extend(dropped)
        trace_all.extend(trace)
    else:
        for qid, parts in by_q.items():
            qt = queries_by_id.get(qid)
            sel, dropped, trace = greedy_select(
                parts,
                cmap,
                token_budget=cfg.selector.token_budget,
                query_text=qt,
            )
            selected_rows.extend(sel)
            for d in dropped:
                d["query_id"] = qid
            dropped_rows.extend(dropped)
            for row in trace:
                row.setdefault("query_id", qid)
            trace_all.extend(trace)

    writer = ArtifactWriter.attach(cfg.run_dir)
    writer.write_jsonl("selected_context.jsonl", selected_rows)
    writer.write_jsonl("dropped_candidates.jsonl", dropped_rows)
    writer.write_jsonl("coverage_trace.jsonl", trace_all)
    return cfg.run_dir
