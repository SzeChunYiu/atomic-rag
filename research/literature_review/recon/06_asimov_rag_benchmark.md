# Asimov RAG benchmark + cross-section retrieval metric

**Status:** stub (P2 deliverable).

## 1. Method one-liner
A synthetic benchmark in which gold passages are inserted at known positions
into clean distractor pools, decomposing end-to-end accuracy into stage-level
efficiencies (chunking × retrieval × selection × generation). A per-method
"cross section" metric isolates retriever performance from corpus and query
luminosity.

## 2. Physics analog (operator-level)
- Asimov dataset (Cowan, Cranmer, Gross, Vitells 2011): synthetic dataset whose
  expected statistical behavior is known by construction.
- Cross-section formalism: $N_{\text{events}} = \sigma \cdot L$, factorizing
  detection rate into intrinsic process probability ($\sigma$) and integrated
  luminosity ($L$).

## 3. Closest prior art
- BEIR (Thakur et al. 2021): broad eval suite.
- LongBench (Bai et al. 2023): long-context evaluation.
- LoTTE (Santhanam et al. 2022): long-tail eval.
- RAG-eval frameworks (RAGAS, ARES): evaluate end-to-end, not stage-decomposed.

## 4. Novelty estimate
- algorithmic: low (synthetic data assembly).
- methodological: high — the stage-decomposition ($\varepsilon_\text{r} \cdot
  \varepsilon_\text{s} \cdot \varepsilon_\text{g}$) is a different way to read
  RAG benchmarks.
- empirical: high if it changes how the field interprets prior results.

## 5. Why publishable
Reviewer-friendly evaluation primitive. Independent NMI/NeurIPS Datasets &
Benchmarks track candidate even before our methods land.

## 6. Falsification protocol
If $\varepsilon_\text{generate}$ already saturates on a benchmark, the
benchmark cannot distinguish retrievers; we discard it from the eval suite
and document why (a positive result for the method too).

## 7. Status
- [ ] gold-injection generator
- [ ] cross-section metric formalization
