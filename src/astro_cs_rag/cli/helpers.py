"""Shared CLI helpers — keep each cli/*.py under the line budget."""

from __future__ import annotations

from pathlib import Path

from astro_cs_rag.config.schema import EmbeddingSettings
from astro_cs_rag.indexing.bm25 import BM25Index
from astro_cs_rag.indexing.dense import DenseIndex
from astro_cs_rag.indexing.embedders import HashEmbedder, SentenceEmbedder
from astro_cs_rag.indexing.io import load_chunks_jsonl, load_index_meta


def embedder_from_settings(settings: EmbeddingSettings) -> HashEmbedder | SentenceEmbedder:
    if settings.use_hash_embedder:
        return HashEmbedder()
    return SentenceEmbedder(settings.model_name, batch_size=settings.batch_size)


def embedder_from_meta(meta: dict[str, object], fallback: EmbeddingSettings) -> HashEmbedder | SentenceEmbedder:
    use_hash = bool(meta.get("use_hash_embedder", fallback.use_hash_embedder))
    if use_hash:
        return HashEmbedder()
    model = str(meta.get("embedding_model", fallback.model_name))
    batch = int(meta.get("batch_size", fallback.batch_size))
    return SentenceEmbedder(model, batch_size=batch)


def load_index_bundle(index_dir: Path) -> tuple[list, BM25Index, DenseIndex, dict[str, object]]:
    chunks = load_chunks_jsonl(index_dir / "chunks.jsonl")
    bm25 = BM25Index.load(index_dir)
    dense = DenseIndex.load(index_dir)
    meta = load_index_meta(index_dir)
    return chunks, bm25, dense, meta


def chunk_map(chunks: list) -> dict[str, object]:
    return {c.chunk_id: c for c in chunks}


def chunk_to_doc_map(chunks: list) -> dict[str, str]:
    return {c.chunk_id: c.doc_id for c in chunks}
