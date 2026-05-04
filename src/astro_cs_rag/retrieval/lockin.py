"""Lock-in coherent paraphrase retrieval.

A lock-in amplifier modulates the input at a known reference frequency,
multiplies by a phase-shifted reference, and low-pass filters; the result
is the in-band component of the signal, with all out-of-band noise rejected.

We adapt this to retrieval:

  1. Generate M paraphrases of a query (the 'modulation').
  2. Retrieve with each → M score fields s_m(i) over chunks.
  3. Build per-chunk per-paraphrase complex amplitudes:
        psi_m(i) = s_m(i) * exp(i * phi_m)
     where phi_m is a learned (or fixed-pattern) per-paraphrase phase.
  4. Aggregate by coherent sum:
        I(i) = | sum_m psi_m(i) |^2.
  5. Compare to the *incoherent* sum sum_m s_m(i)^2; the difference is the
     phase-coherent gain. The headline empirical claim is that I(i) > the
     incoherent baseline for true evidence and ~equal for distractors.

This module accepts pre-computed score fields per paraphrase; the actual
paraphrase generation lives in `generation/paraphrase.py`.
"""

from __future__ import annotations

import cmath
import math

import numpy as np


def coherent_sum(
    score_fields: list[dict[str, float]],
    *,
    phases: list[float] | None = None,
) -> dict[str, float]:
    """Phase-coherent aggregation across M score fields.

    `phases` is a length-M list of per-paraphrase phases (radians). If None,
    use phi_m = 0 (purely constructive — i.e. the simplest coherent variant
    that still differs from incoherent because we sum amplitudes rather than
    intensities).
    """
    if not score_fields:
        return {}
    M = len(score_fields)
    if phases is None:
        phases = [0.0] * M
    if len(phases) != M:
        msg = "phases must align with score_fields"
        raise ValueError(msg)
    chunk_ids: list[str] = list({cid for sf in score_fields for cid in sf})
    if not chunk_ids:
        return {}
    # Vectorize: build an (M, K) score matrix and apply phases via complex mul.
    score_mat = np.zeros((M, len(chunk_ids)), dtype=np.float64)
    for m, sf in enumerate(score_fields):
        for k, cid in enumerate(chunk_ids):
            v = sf.get(cid)
            if v is not None:
                score_mat[m, k] = float(v)
    phase_factors = np.exp(1j * np.asarray(phases, dtype=np.float64))
    z = phase_factors @ score_mat  # shape (K,)
    intens = np.abs(z) ** 2
    return {cid: float(intens[k]) for k, cid in enumerate(chunk_ids)}


def incoherent_sum(score_fields: list[dict[str, float]]) -> dict[str, float]:
    """Sum of squared scores (intensities) — the baseline ensemble."""
    if not score_fields:
        return {}
    chunk_ids: list[str] = list({cid for sf in score_fields for cid in sf})
    if not chunk_ids:
        return {}
    # Vectorize sum of squared scores across paraphrases.
    score_mat = np.zeros((len(score_fields), len(chunk_ids)), dtype=np.float64)
    for m, sf in enumerate(score_fields):
        for k, cid in enumerate(chunk_ids):
            v = sf.get(cid)
            if v is not None:
                score_mat[m, k] = float(v)
    intens = np.sum(score_mat * score_mat, axis=0)
    return {cid: float(intens[k]) for k, cid in enumerate(chunk_ids)}


def coherence_ratio(
    coherent: dict[str, float],
    incoherent: dict[str, float],
) -> dict[str, float]:
    """Per-chunk coherence ratio I_coh / I_incoh — measures phase information."""
    out: dict[str, float] = {}
    for cid in incoherent:
        i = float(incoherent[cid])
        if i <= 1e-12:
            out[cid] = 0.0
            continue
        out[cid] = float(coherent.get(cid, 0.0)) / i
    return out


def fixed_pattern_phases(M: int) -> list[float]:
    """Constructive phase pattern: all paraphrases share phase 0.

    Lock-in physics: the reference oscillator has ONE phase. Paraphrases are
    independent realizations of the same query intent; if a chunk is true
    evidence, all paraphrases score it similarly and should add coherently.
    With phi_m = 0 for all m: |sum s_m|^2 = (sum s_m)^2 grows like M^2 for
    invariant signal, while noise (random sign across paraphrases) only
    grows like M, giving an SNR boost.

    The earlier implementation returned equally spaced phases on the full
    unit circle. With M paraphrases that produce the same score, the sum
    of M roots of unity is identically zero — destructive interference,
    the opposite of the claimed "coherent gain". This was a conceptual
    bug that would have invalidated Theorem 2 (encoder-noise rejection).
    """
    if M <= 0:
        return []
    return [0.0] * M


def equally_spaced_phases(M: int) -> list[float]:
    """Diagnostic-only: equally spaced phases on the unit circle.

    NOT a coherent aggregator — sum of M roots of unity is 0 for invariant
    signal. Provided only for ablation/diagnostic experiments where the
    contrast (constructive vs destructive sum) is itself the measurement.
    """
    if M <= 0:
        return []
    return [2.0 * math.pi * m / M for m in range(M)]
