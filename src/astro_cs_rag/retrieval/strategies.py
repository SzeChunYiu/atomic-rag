"""Per-mode retrieval scoring strategies.

Each strategy consumes the loaded index bundle, the query text and embedding,
and returns a fused score dict (chunk_id -> score) ready for `build_candidates`.
Strategies are pure functions so they can be ablated cheanly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astro_cs_rag.indexing.bm25 import BM25Index
from astro_cs_rag.indexing.dense import DenseIndex
from astro_cs_rag.retrieval.fusion import rank_by_score, reciprocal_rank_fusion


def fusion_rrf_scores(
    bm25: BM25Index, dense: DenseIndex, query_text: str, q_emb: np.ndarray
) -> dict[str, float]:
    b_scores = bm25.scores(query_text)
    d_scores = dense.scores(q_emb)
    r1 = rank_by_score(b_scores)
    r2 = rank_by_score(d_scores)
    return reciprocal_rank_fusion([r1, r2])


def bm25_scores(bm25: BM25Index, query_text: str) -> dict[str, float]:
    return bm25.scores(query_text)


def dense_scores(dense: DenseIndex, q_emb: np.ndarray) -> dict[str, float]:
    return dense.scores(q_emb)


def late_interaction_scores(
    index_dir: Path,
    query_text: str,
    *,
    use_hash_encoder: bool = False,
    encoder_model_name: str | None = None,
) -> dict[str, float]:
    from astro_cs_rag.indexing.multivec import MultiVecIndex
    from astro_cs_rag.indexing.multivec_encoder import (
        HashMultiVecEncoder,
        STMultiVecEncoder,
    )

    idx = MultiVecIndex.load(index_dir)
    if use_hash_encoder or encoder_model_name is None:
        enc: HashMultiVecEncoder | STMultiVecEncoder = HashMultiVecEncoder()
    else:
        enc = STMultiVecEncoder(encoder_model_name)
    q_tokens = enc.encode_tokens([query_text])[0]
    return idx.maxsim_scores(q_tokens)


def hierarchical_scores(
    index_dir: Path,
    q_emb: np.ndarray,
) -> dict[str, float]:
    from astro_cs_rag.indexing.hierarchical import hierarchy_scores, load_hierarchy

    nodes = load_hierarchy(index_dir)
    return hierarchy_scores(nodes, q_emb, flatten_to_chunks=True)


def lockin_scores(
    bm25: BM25Index,
    dense: DenseIndex,
    embedder,
    query_text: str,
    *,
    n_paraphrases: int = 4,
    use_fixed_pattern_phase: bool = True,
    paraphrase_settings=None,
    paraphrases: list[str] | None = None,
) -> dict[str, float]:
    """Lock-in coherent paraphrase retrieval.

    Paraphrase source priority:
      1. ``paraphrases`` (pre-cached list): use as-is, ignore other params.
      2. ``paraphrase_settings`` (LLM): generate at runtime.
      3. fall back to ``[query_text]`` (no paraphrases).
    """
    from astro_cs_rag.retrieval.lockin import coherent_sum, fixed_pattern_phases

    if paraphrases is not None:
        paras = list(paraphrases)
    elif paraphrase_settings is not None:
        from astro_cs_rag.generation.paraphrase import generate_paraphrases
        paras = generate_paraphrases(query_text, n=n_paraphrases, settings=paraphrase_settings)
    else:
        paras = [query_text]

    fields: list[dict[str, float]] = []
    for p in paras:
        q_emb = embedder.encode([p])[0]
        sf = fusion_rrf_scores(bm25, dense, p, q_emb)
        fields.append(sf)
    phases = fixed_pattern_phases(len(fields)) if use_fixed_pattern_phase else None
    return coherent_sum(fields, phases=phases)


def splade_scores(
    index_dir: Path,
    query_text: str,
    *,
    use_hash_backend: bool = False,
    model_name: str = "naver/splade-v3",
) -> dict[str, float]:
    from astro_cs_rag.indexing.splade import HashSpladeBackend, HFSpladeBackend, SpladeIndex

    idx = SpladeIndex(
        chunk_ids=_load_splade_chunk_ids(index_dir),
        vectors=np.load(index_dir / "splade_vectors.npy"),
        backend_name=str((index_dir / "splade_backend.txt").read_text(encoding="utf-8")).strip(),
    )
    backend: HashSpladeBackend | HFSpladeBackend = (
        HashSpladeBackend() if use_hash_backend else HFSpladeBackend(model_name=model_name)
    )
    q_vec = backend.encode([query_text])[0]
    return idx.scores(q_vec)


def _load_splade_chunk_ids(index_dir: Path) -> list[str]:
    import json

    path = index_dir / "splade_chunk_ids.json"
    return list(json.loads(path.read_text(encoding="utf-8")))
