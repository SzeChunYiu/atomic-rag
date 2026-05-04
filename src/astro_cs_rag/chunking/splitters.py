"""Character-based chunking with overlap and stable offsets."""

from __future__ import annotations

from astro_cs_rag.atoms.schemas import Chunk, Document


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be in [0, chunk_size)"
        raise ValueError(msg)

    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text
        if not text.strip():
            continue
        start = 0
        part = 0
        stride = chunk_size - chunk_overlap
        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end]
            if piece.strip():
                cid = f"{doc.doc_id}::{part}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        text=piece,
                        start_char=start,
                        end_char=end,
                        token_count=_estimate_tokens(piece),
                        metadata=dict(doc.metadata),
                    )
                )
                part += 1
            if end >= len(text):
                break
            start += stride
    return chunks


def write_chunk_manifest(path: str, chunks: list[Chunk]) -> None:
    """JSON manifest for chunk inventory (used by benchmarks)."""
    import json
    from pathlib import Path

    rows = [c.model_dump(mode="json") for c in chunks]
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
