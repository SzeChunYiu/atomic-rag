"""Query out-of-distribution gate.

Approximate the full 'CATHODE/ANODE' normalizing-flow approach with a
diagonal-covariance Gaussian baseline: fit a multivariate normal to the
query-embedding distribution at calibration time; at inference flag any
query whose log-likelihood falls below the alpha-quantile of the training
log-likelihoods. This is a cheap, calibrated OOD detector — sufficient as a
diagnostic and as a routing signal for adaptive retrieval recipes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianOODModel:
    mean: np.ndarray
    var: np.ndarray
    threshold_logp: float

    def log_likelihood(self, x: np.ndarray) -> float:
        diff = x - self.mean
        return float(-0.5 * np.sum(diff * diff / (self.var + 1e-9)))

    def is_ood(self, x: np.ndarray) -> bool:
        return self.log_likelihood(x) < self.threshold_logp


def fit_gaussian_ood(
    train_embeddings: np.ndarray,
    *,
    alpha: float = 0.05,
) -> GaussianOODModel:
    if train_embeddings.size == 0:
        return GaussianOODModel(
            mean=np.zeros((1,), dtype=np.float32),
            var=np.ones((1,), dtype=np.float32),
            threshold_logp=-1e9,
        )
    mu = train_embeddings.mean(axis=0)
    var = train_embeddings.var(axis=0) + 1e-6
    diffs = train_embeddings - mu
    logps = -0.5 * np.sum(diffs * diffs / (var + 1e-9), axis=1)
    threshold = float(np.quantile(logps, alpha))
    return GaussianOODModel(mean=mu.astype(np.float32), var=var.astype(np.float32), threshold_logp=threshold)
