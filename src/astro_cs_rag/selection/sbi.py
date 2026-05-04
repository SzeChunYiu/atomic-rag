"""Simulation-based-inference posterior over evidence sets.

Faithful sketch of the neural-posterior-estimation approach:

  1. Simulate (q, E, a) triples by sampling subsets E of candidates and
     using the generator to produce a from (q, E). Score each by held-out
     answer-quality (EM/F1).
  2. Train a small neural ratio estimator log p(E | q, a) that takes feature
     vectors of E and returns a calibrated posterior weight.
  3. At inference, score each candidate subset E by the trained estimator;
     pick argmax (or top-k posterior mass) under token budget.

The full implementation lives in P7. This module ships the *simulator* +
*feature extractor* now so the SBI training pipeline has a stable interface.
A lightweight stand-in selector (uniform-random over subsets up to budget,
ranked by mean SNR) is provided for end-to-end smoke runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CandidateSummary:
    chunk_id: str
    snr: float
    token_count: int
    embedding: np.ndarray | None


def featurize_subset(subset: Sequence[CandidateSummary]) -> np.ndarray:
    """A small, fixed-dim feature vector summarizing an evidence subset.

    Features (10-D):
      [|E|, sum_snr, mean_snr, max_snr, min_snr, total_tokens,
       mean_pairwise_cosine, max_pairwise_cosine, std_snr, count_above_snr2]
    """
    if not subset:
        return np.zeros((10,), dtype=np.float32)
    snrs = np.asarray([float(c.snr) for c in subset], dtype=np.float32)
    tokens = float(sum(c.token_count for c in subset))
    feats = [
        float(len(subset)),
        float(snrs.sum()),
        float(snrs.mean()),
        float(snrs.max()),
        float(snrs.min()),
        tokens,
    ]
    embs = [c.embedding for c in subset if c.embedding is not None]
    if len(embs) >= 2:
        E = np.stack(embs, axis=0)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = E @ E.T
        iu = np.triu_indices_from(sims, k=1)
        feats.append(float(sims[iu].mean()))
        feats.append(float(sims[iu].max()))
    else:
        feats.extend([0.0, 0.0])
    feats.append(float(snrs.std()))
    feats.append(float((snrs > 2.0).sum()))
    return np.asarray(feats, dtype=np.float32)


def simulate_subsets(
    candidates: list[CandidateSummary],
    *,
    n_simulations: int,
    token_budget: int,
    seed: int = 0,
) -> list[list[CandidateSummary]]:
    """Sample evidence subsets by uniform random selection under budget.

    Returns a list of subsets; downstream code pairs each with an answer
    score to train the SBI posterior.
    """
    if not candidates:
        return []
    rng = random.Random(seed)
    out: list[list[CandidateSummary]] = []
    for _ in range(n_simulations):
        order = list(range(len(candidates)))
        rng.shuffle(order)
        subset: list[CandidateSummary] = []
        used = 0
        for i in order:
            c = candidates[i]
            if used + c.token_count > token_budget:
                continue
            subset.append(c)
            used += c.token_count
        out.append(subset)
    return out


def stand_in_posterior_score(subset: list[CandidateSummary]) -> float:
    """Replaces the trained NPE posterior with a hand-crafted reward.

    Used during P5 development before the NPE network is trained. Should be
    monotone in mean_snr and decreasing in token_count to mimic Pareto.
    """
    feat = featurize_subset(subset)
    if feat[0] == 0:
        return -float("inf")
    mean_snr = feat[2]
    tokens = feat[5]
    return float(mean_snr - 1e-3 * tokens)
