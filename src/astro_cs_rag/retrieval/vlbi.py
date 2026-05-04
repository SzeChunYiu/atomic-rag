"""Cross-correlate score fields from independent embedders ('VLBI' fusion).

Very-long-baseline interferometry combines signals from spatially separated
telescopes; their cross-correlation yields effective angular resolution far
beyond any single station. Analog:

  - Run dense retrieval with M independent embedders (BGE-M3, E5, GTE, ...).
  - For each chunk, compute the cross-correlation of its M score vectors
    against the M query vectors. Chunks whose score is reproducibly high
    across embedders are 'real'; chunks high in one but not others are
    'artifacts' of that embedder's bias.
  - Combine via the geometric mean of (score - background_per_embedder),
    rectifying inconsistent signals.
"""

from __future__ import annotations

import math


def vlbi_combine(
    score_fields: list[dict[str, float]],
    *,
    rectify: bool = True,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Combine M dict-of-scores into a single dict via VLBI-style aggregation.

    score_fields[m][chunk_id] = score from embedder m. Returns chunk_id ->
    combined score = geometric mean of normalized positive scores.
    """
    if not score_fields:
        return {}
    chunk_ids: set[str] = set()
    for sf in score_fields:
        chunk_ids.update(sf.keys())

    out: dict[str, float] = {}
    for cid in chunk_ids:
        accum = 0.0
        n = 0
        for sf in score_fields:
            s = float(sf.get(cid, 0.0))
            if rectify:
                s = max(s, 0.0)
            accum += math.log(s + eps)
            n += 1
        if n == 0:
            continue
        out[cid] = float(math.exp(accum / n))
    return out


def vlbi_visibility(
    score_fields: list[dict[str, float]],
    *,
    chunk_id: str,
) -> float:
    """A chunk's 'fringe visibility' — coherence of its M scores.

    visibility = 1 - std/mean over M, clipped to [0, 1].
    Visibility ≈ 1 → all embedders agree (real signal); ≈ 0 → noise.
    """
    if not score_fields:
        return 0.0
    vals = [max(0.0, float(sf.get(chunk_id, 0.0))) for sf in score_fields]
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    if m <= 1e-12:
        return 0.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    sigma = var ** 0.5
    return max(0.0, min(1.0, 1.0 - sigma / max(m, 1e-9)))
