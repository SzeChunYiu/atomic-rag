"""Build optional side-indices alongside the BM25+dense bundle.

Each function is a no-op unless the corresponding setting is enabled in the
benchmark config. Materializes per-mode artifacts under the same `index_dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from astro_cs_rag.atoms.schemas import Chunk


def build_late_interaction_side_index(
    index_dir: Path,
    chunks: list[Chunk],
    *,
    use_hash_encoder: bool,
    encoder_model_name: str,
) -> None:
    from astro_cs_rag.indexing.multivec import MultiVecIndex
    from astro_cs_rag.indexing.multivec_encoder import (
        HashMultiVecEncoder,
        STMultiVecEncoder,
    )

    enc: HashMultiVecEncoder | STMultiVecEncoder = (
        HashMultiVecEncoder() if use_hash_encoder else STMultiVecEncoder(encoder_model_name)
    )
    idx = MultiVecIndex.from_texts(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        encoder=enc,
    )
    idx.save(index_dir)


def build_hierarchical_side_index(
    index_dir: Path,
    chunks: list[Chunk],
    leaf_embeddings: np.ndarray,
    *,
    branching: int = 6,
    max_levels: int = 4,
) -> None:
    from astro_cs_rag.indexing.hierarchical import build_hierarchy, save_hierarchy

    nodes = build_hierarchy(
        leaf_chunk_ids=[c.chunk_id for c in chunks],
        leaf_texts=[c.text for c in chunks],
        leaf_embeddings=leaf_embeddings,
        branching=branching,
        max_levels=max_levels,
    )
    save_hierarchy(nodes, index_dir)


def build_splade_side_index(
    index_dir: Path,
    chunks: list[Chunk],
    *,
    use_hash_backend: bool,
    model_name: str,
) -> None:
    from astro_cs_rag.indexing.splade import HashSpladeBackend, HFSpladeBackend, SpladeIndex

    backend = HashSpladeBackend() if use_hash_backend else HFSpladeBackend(model_name=model_name)
    idx = SpladeIndex.from_texts(
        chunk_ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        backend=backend,
    )
    np.save(index_dir / "splade_vectors.npy", idx.vectors)
    (index_dir / "splade_chunk_ids.json").write_text(
        json.dumps(idx.chunk_ids), encoding="utf-8"
    )
    (index_dir / "splade_backend.txt").write_text(idx.backend_name, encoding="utf-8")
