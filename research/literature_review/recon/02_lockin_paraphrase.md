# Lock-in coherent paraphrase retrieval

**Status:** stub (will be filled in P1 reconnaissance week).

## 1. Method one-liner
Generate $M$ paraphrases of the query, retrieve with each, score chunks by the
*coherent* sum $|\sum_m e^{i\phi_m} s_m(i)|^2$ across paraphrases. True evidence
is in-phase across paraphrases; embedding artifacts are out-of-phase.

## 2. Physics analog (operator-level)
Lock-in amplifier / homodyne detection. Modulate the signal at a known reference
frequency, demodulate and low-pass filter, recover signal at the modulation
band. Replaces shot-noise-limited detection with phase-sensitive detection.

## 3. Closest prior art (preliminary)
- HyDE (Gao et al. 2022): generates a hypothetical document, retrieves against
  it. Single-pass, no phase concept.
- Query2doc (Wang et al. 2023): query expansion via LLM-generated pseudo-doc.
- Multi-query / fusion-in-decoder retrievers: ensemble across paraphrases by
  *averaging* — the "incoherent sum" baseline we explicitly compare against.

## 4. Novelty estimate
- algorithmic: high (phase-aware aggregation across paraphrases).
- theoretical: medium (phase as a learned per-paraphrase rotation).
- empirical: high (the coherent-vs-incoherent gap is a quantification of
  encoder noise temperature — a publishable diagnostic on its own).

## 5. Why publishable
Attacks F1 (topical-but-wrong) and F3 (query degeneracy). The coherent-minus-
incoherent gap is itself a measurement of encoder-induced semantic noise — a
quantity the field has speculated about but never quantified.

## 6. Falsification protocol
If coherent ≤ incoherent ensemble (within $p<0.01$) on all benchmarks, the
phase concept does no work. Publishable as a null result on phase information
in dense retrieval.

## 7. Status
- [ ] prior-art search done
- [ ] minimal implementation sketch
- [ ] ablation plan written
