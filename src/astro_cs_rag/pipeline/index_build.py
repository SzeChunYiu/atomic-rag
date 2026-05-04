"""Build persisted BM25 + dense bundle."""

from __future__ import annotations

from pathlib import Path

from astro_cs_rag.chunking.splitters import chunk_documents
from astro_cs_rag.config.schema import IndexConfig
from astro_cs_rag.data.loaders import load_corpus_jsonl
from astro_cs_rag.indexing.bm25 import BM25Index
from astro_cs_rag.indexing.dense import DenseIndex
from astro_cs_rag.indexing.embedders import HashEmbedder, SentenceEmbedder
from astro_cs_rag.indexing.digests import model_revision_digest
from astro_cs_rag.indexing.io import save_chunks_jsonl, save_index_meta
from astro_cs_rag.pipeline.side_indices import (
    build_hierarchical_side_index,
    build_late_interaction_side_index,
    build_splade_side_index,
)


def _embedder(cfg: IndexConfig) -> HashEmbedder | SentenceEmbedder:
    if cfg.embedding.use_hash_embedder:
        return HashEmbedder()
    return SentenceEmbedder(
        cfg.embedding.model_name, batch_size=cfg.embedding.batch_size
    )


def build_index_bundle(cfg: IndexConfig) -> Path:
    index_dir = cfg.paths.output_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cfg.paths.corpus_path
    if corpus_path is None:
        msg = "corpus_path required"
        raise ValueError(msg)

    # Reuse-existing-index check: if a complete bundle for this corpus +
    # chunk geometry + embedder already exists, skip the rebuild. Massive
    # win when sweeping selectors at fixed (corpus, chunk_size, embedder):
    # the index would otherwise be rebuilt N times (BGE-M3 embedding
    # 8000 chunks ≈ 3–5 min each).
    meta_path = index_dir / "index_meta.json"
    if meta_path.is_file() and (index_dir / "embeddings.npy").is_file() and (index_dir / "chunks.jsonl").is_file():
        try:
            import json as _json
            existing = _json.loads(meta_path.read_text(encoding="utf-8"))
            same = (
                existing.get("chunk_size") == int(cfg.chunk_size)
                and existing.get("chunk_overlap") == int(cfg.chunk_overlap)
                and existing.get("embedding_model") == cfg.embedding.model_name
                and bool(existing.get("use_hash_embedder", False)) == bool(cfg.embedding.use_hash_embedder)
            )
            if same:
                return index_dir
        except (OSError, ValueError, KeyError):
            pass

    docs = load_corpus_jsonl(corpus_path)
    chunks = chunk_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        msg = "No chunks produced — check corpus content"
        raise ValueError(msg)

    bm25 = BM25Index.from_chunks(chunks)
    bm25.save(index_dir)

    embedder = _embedder(cfg)
    texts = [c.text for c in chunks]
    emb = embedder.encode(texts)
    dense = DenseIndex([c.chunk_id for c in chunks], emb)
    dense.save(index_dir)

    save_chunks_jsonl(chunks, index_dir / "chunks.jsonl")
    digest: dict[str, object] = {}
    if not cfg.embedding.use_hash_embedder:
        digest = model_revision_digest(cfg.embedding.model_name)
    meta: dict[str, object] = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "embedding_model": cfg.embedding.model_name,
        "embedding_digest": digest,
        "use_hash_embedder": cfg.embedding.use_hash_embedder,
        "batch_size": cfg.embedding.batch_size,
        "chunk_count": len(chunks),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
    }
    save_index_meta(index_dir, meta)

    if cfg.side_indices.late_interaction:
        build_late_interaction_side_index(
            index_dir,
            chunks,
            use_hash_encoder=cfg.side_indices.late_interaction_use_hash,
            encoder_model_name=cfg.side_indices.late_interaction_model,
        )
    if cfg.side_indices.hierarchical:
        build_hierarchical_side_index(
            index_dir,
            chunks,
            leaf_embeddings=emb,
            branching=cfg.side_indices.hierarchical_branching,
            max_levels=cfg.side_indices.hierarchical_max_levels,
        )
    if cfg.side_indices.splade:
        build_splade_side_index(
            index_dir,
            chunks,
            use_hash_backend=cfg.side_indices.splade_use_hash,
            model_name=cfg.side_indices.splade_model,
        )
    return index_dir
