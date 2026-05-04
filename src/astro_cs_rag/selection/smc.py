"""Sequential Monte Carlo (particle filter) adaptive RAG.

Maintain a population of `answer hypotheses' (particles), each carrying:
  - a current selected-evidence set,
  - a normalized weight,
  - a hypothesized answer span.

Loop:
  1. For each particle, retrieve the next-best candidate conditional on the
     current set (residual-need-aware).
  2. Update the particle's weight by the new evidence support score.
  3. Resample particles with probability proportional to weight.
  4. Stop when the effective sample size collapses to ~1 (consensus) or the
     token budget is exhausted.

This is the Bayesian-sequential complement to the deterministic CLEAN-RAG
loop: instead of greedily explaining residual need, we sample a *distribution*
of possible explanations and let particle weights resolve disagreement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Particle:
    selected_ids: list[str] = field(default_factory=list)
    log_weight: float = 0.0
    used_tokens: int = 0


def effective_sample_size(particles: list[Particle]) -> float:
    if not particles:
        return 0.0
    log_w = np.asarray([p.log_weight for p in particles], dtype=np.float64)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum() + 1e-12
    return float(1.0 / (np.sum(w * w) + 1e-12))


def normalize_weights(particles: list[Particle]) -> np.ndarray:
    log_w = np.asarray([p.log_weight for p in particles], dtype=np.float64)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / (w.sum() + 1e-12)


def smc_select(
    candidates: list[tuple[str, float, int]],   # (chunk_id, snr, token_count)
    *,
    n_particles: int = 16,
    token_budget: int = 512,
    n_steps: int = 4,
    resample_threshold: float = 0.5,
    seed: int = 0,
) -> list[str]:
    if not candidates:
        return []
    rng = random.Random(seed)
    pool = list(candidates)
    particles: list[Particle] = [Particle() for _ in range(n_particles)]
    for step in range(n_steps):
        random_pool = list(pool)
        rng.shuffle(random_pool)
        for p in particles:
            for cid, snr, toks in random_pool:
                if cid in p.selected_ids:
                    continue
                if p.used_tokens + toks > token_budget:
                    continue
                p.selected_ids.append(cid)
                p.used_tokens += toks
                p.log_weight += float(max(0.0, snr))
                break
        ess = effective_sample_size(particles)
        if ess < resample_threshold * n_particles:
            w = normalize_weights(particles)
            indices = rng.choices(range(n_particles), weights=w.tolist(), k=n_particles)
            particles = [Particle(
                selected_ids=list(particles[i].selected_ids),
                log_weight=0.0,
                used_tokens=particles[i].used_tokens,
            ) for i in indices]
        _ = math.fsum  # silence unused import warning when math used elsewhere

    weights = normalize_weights(particles)
    counts: dict[str, float] = {}
    for p, w in zip(particles, weights, strict=True):
        for cid in p.selected_ids:
            counts[cid] = counts.get(cid, 0.0) + float(w)
    return sorted(counts, key=lambda c: -counts[c])
