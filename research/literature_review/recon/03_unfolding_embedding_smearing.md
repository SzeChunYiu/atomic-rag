# D'Agostini unfolding of embedding smearing

**Status:** stub.

## 1. Method one-liner
Treat the dense retrieval score field as a *smeared* image of the true
relevance distribution over claim clusters. Solve for the underlying
distribution via iterative Bayesian (D'Agostini) unfolding, then re-rank.

## 2. Physics analog (operator-level)
Iterative Bayesian unfolding (D'Agostini 1995, NIM A 362:487) — invert a
known response matrix $R_{ij}$ that maps true bin $j$ to observed bin $i$.
Standard tool in HEP detector deconvolution.

## 3. Closest prior art (preliminary)
- HyDE / Query2doc: hallucinate target document, retrieve against it.
- Pseudo-relevance feedback (Rocchio, RM3): expand query using top retrieved
  passages.
- Embedding inversion attacks (Morris et al. 2023): different goal.
None solves an inverse problem with a learned response operator.

## 4. Novelty estimate
- algorithmic: high.
- theoretical: high (inverse problem framing of retrieval).
- empirical: high (response matrix as a per-corpus learned object).

## 5. Why publishable
Attacks F3 (query degeneracy) with a principled estimator. NMI/NeurIPS-shaped
because the response matrix $R$ becomes a *new* learned object in IR — open
research direction beyond the immediate paper.

## 6. Falsification protocol
If unfolding does not exceed RM3 expansion + dense retrieval, we re-position
as a diagnostic of embedding smearing, not a method.

## 7. Status
- [ ] prior-art search done
- [ ] minimal implementation sketch
