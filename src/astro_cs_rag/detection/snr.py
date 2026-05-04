"""Evidence SNR from retrieval scores vs background (tail or global)."""

from __future__ import annotations

import math
from typing import Literal

from astro_cs_rag.atoms.schemas import EvidenceAtom
from astro_cs_rag.detection.background import global_mean_std, tail_mean_std


def detect_evidence(
    query_id: str,
    scores: dict[str, float],
    *,
    window: int = 10,
    snr_threshold: float = 0.0,
    background_mode: Literal["tail", "global"] = "tail",
    chunk_texts: dict[str, str] | None = None,
    length_normalize_snr: bool = False,
) -> list[EvidenceAtom]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    vals = [v for _, v in items]
    if not vals:
        return []

    if background_mode == "global":
        mu, sigma = global_mean_std(vals)
    else:
        mu, sigma = tail_mean_std(vals, window)
    denom = sigma + 1e-9

    atoms: list[EvidenceAtom] = []
    for rank, (cid, raw) in enumerate(items, start=1):
        snr = (float(raw) - mu) / denom
        if length_normalize_snr and chunk_texts:
            ln = len(chunk_texts.get(cid, ""))
            snr = snr / (math.sqrt(float(ln)) + 1e-9)
        atoms.append(
            EvidenceAtom(
                query_id=query_id,
                chunk_id=cid,
                raw_score=float(raw),
                bg_mean=mu,
                bg_std=sigma,
                snr=float(snr),
                detector_rank=rank,
                metadata={
                    "tail_window": window,
                    "background_mode": background_mode,
                    "length_normalize_snr": length_normalize_snr,
                },
            )
        )

    if snr_threshold > 0.0:
        filtered = [a for a in atoms if a.snr >= snr_threshold]
        atoms = [
            a.model_copy(update={"detector_rank": i})
            for i, a in enumerate(filtered, start=1)
        ]

    return atoms
