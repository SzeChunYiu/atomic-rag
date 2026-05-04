"""Aperture-photometry SNR: cosine-ball local background instead of rank-tail.

Aperture photometry in astronomy measures the flux in a circular aperture
around a source and subtracts the mean flux of an annulus or sky region
*at fixed angular radius*. Translated to retrieval:

  background of candidate i = mean / std of scores of chunks that lie within
  cosine radius (radius_in, radius_out] of chunk i in embedding space.

This is more principled than the existing tail-window background because the
neighborhood is defined geometrically, not by rank.
"""

from __future__ import annotations

import numpy as np

from astro_cs_rag.atoms.schemas import EvidenceAtom


def aperture_snr(
    *,
    query_id: str,
    candidate_chunk_ids: list[str],
    candidate_scores: dict[str, float],
    embeddings_by_id: dict[str, np.ndarray],
    radius_in: float = 0.10,
    radius_out: float = 0.50,
    snr_threshold: float = 0.0,
    eps: float = 1e-9,
) -> list[EvidenceAtom]:
    """Compute z-score relative to the cosine-annulus around each candidate.

    radius_* are *cosine distances* in [0, 2]. Defaults: 0.10–0.50 forms an
    annulus that excludes the source itself and its closest near-duplicates.
    """
    items = sorted(candidate_scores.items(), key=lambda x: -x[1])
    cids = [c for c, _ in items]
    scores = np.asarray([candidate_scores[c] for c in cids], dtype=np.float64)

    embs = np.stack(
        [embeddings_by_id[c] / (np.linalg.norm(embeddings_by_id[c]) + eps) for c in cids],
        axis=0,
    ).astype(np.float32)
    sim = embs @ embs.T              # cosine similarity matrix
    cos_dist = np.clip(1.0 - sim, 0.0, 2.0)

    atoms: list[EvidenceAtom] = []
    for i, cid in enumerate(cids):
        ring_mask = (cos_dist[i] >= radius_in) & (cos_dist[i] <= radius_out)
        if not bool(ring_mask.any()):
            mu, sigma = float(np.mean(scores)), float(np.std(scores))
        else:
            ring_scores = scores[ring_mask]
            mu = float(np.mean(ring_scores))
            sigma = float(np.std(ring_scores))
        z = (float(scores[i]) - mu) / (sigma + eps)
        atoms.append(
            EvidenceAtom(
                query_id=query_id,
                chunk_id=cid,
                raw_score=float(scores[i]),
                bg_mean=mu,
                bg_std=sigma,
                snr=float(z),
                detector_rank=i + 1,
                metadata={
                    "background_mode": "aperture",
                    "radius_in": radius_in,
                    "radius_out": radius_out,
                    "n_ring": int(ring_mask.sum()),
                },
            )
        )
    if snr_threshold > 0.0:
        atoms = [a for a in atoms if a.snr >= snr_threshold]
        atoms = [a.model_copy(update={"detector_rank": i + 1}) for i, a in enumerate(atoms)]
    return atoms
