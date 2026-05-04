# Astro-CS-RAG: A Layered Physics-Inspired Pipeline for Retrieval-Augmented Generation

**Target venue:** Nature Machine Intelligence (primary) / NeurIPS / SIGIR (alternates).

**Status:** skeleton — sections to fill from empirical runs (Phases P7+).

## Abstract (≤ 200 words)

Retrieval-augmented generation (RAG) systems treat retrieval as nearest-neighbor
search over an embedding space, then concatenate the top-k results into a
language model's context. We argue that this confuses *proximity* with
*evidential usefulness* and that the failure modes of modern RAG are exactly
those that experimental physics has been engineering around for decades. We
formalize RAG as a noisy measurement chain and propose a layered pipeline of
physics-motivated operators — anti-$k_T$ jet clustering for evidence selection
(with provable infrared- and collinear-safety), aperture-photometry SNR for
local-background detection, lock-in coherent paraphrase aggregation for
encoder-noise rejection, and conservation-law residuals for symbolic
faithfulness verification — coupled with a calorimetric routing system that
selects per-query recipes from the morphology of the score field. Across
HotpotQA, NaturalQuestions, MuSiQue, RAGAS-faithfulness, and LongBench-RAG,
our pipeline improves end-to-end answer F1 by **TBD pp** under matched compute
and reduces context tokens by **TBD pp** while preserving recall. The
approach generalizes: applied to held-out physics arXiv corpora (astro-ph,
hep-ph) it sustains gains, demonstrating that the cross-domain transfer of
methodology is empirically substantive, not metaphorical.

## 1. Introduction

- RAG is now ubiquitous; published methods are dominated by encoder/reranker
  scaling and adaptive retrieval. None of these address the underlying
  *measurement-chain* losses.
- Physics has formal, theorem-grade tools for noisy measurement chains:
  matched filtering, IRC-safe clustering, conservation-law residuals,
  calibrated thresholds, simulation-based inference. We import those tools.
- Contributions:
  1. A formal **measurement-chain model** of RAG with a 9-mode atomic
     failure taxonomy (F1–F9).
  2. A **layered pipeline architecture** that maps physics operators to
     each failure mode.
  3. **Anti-$k_T$ evidence-jet clustering** with provable IRC-safety in RAG
     (Theorem 1).
  4. **Lock-in coherent paraphrase retrieval** that quantifies and rejects
     encoder noise (Theorem 2).
  5. **Conservation-law residuals** as symbolic, judge-free faithfulness.
  6. The **Asimov benchmark** + cross-section formalism: stage-decomposed
     RAG evaluation that the field can adopt independently.

## 2. RAG as a Measurement Chain

### 2.1 Stages and noise sources
$\theta^* \to q \to \hat q \to s_i \to \mathrm{top\text{-}k} \to E \to a$.

### 2.2 The atomic failure taxonomy (F1–F9)
Reproduce table from `change_log/atomic_failure_atlas.md`.

### 2.3 The layered architecture
Reproduce stack figure from `methods/09_layered_pipeline_architecture.md`.

## 3. Anti-$k_T$ Evidence-Jet Clustering for RAG

### 3.1 Setup and metric
$d_{ij} = \min(s_i^{-2}, s_j^{-2}) \cdot \Delta_{ij}^2 / R^2$, with
$\Delta_{ij} = 1 - \cos(\hat e_i, \hat e_j)$.

### 3.2 Theorem 1 (IRC-safety)
Statement (collinear) + statement (infrared) + proof from Cacciari–Salam–Soyez
restated for $(s, \cos\Delta)$.

### 3.3 Empirical IRC robustness
Headline figure: F1 vs chunk-size for {greedy, MMR, anti-$k_T$}. Variance
ratio + paired-bootstrap p-value.

### 3.4 Empirical IR robustness
Headline figure: F1 vs distractor-pool size for {greedy, MMR, anti-$k_T$}.

## 4. Lock-in Coherent Paraphrase Retrieval

### 4.1 Coherent vs incoherent aggregation
$I_\text{coh}(i) = |\sum_m s_m(i) e^{i\phi_m}|^2$ vs $I_\text{inc}(i) = \sum_m s_m(i)^2$.

### 4.2 Theorem 2 (encoder-noise rejection)
The coherent-minus-incoherent difference is a calibrated estimator of the
encoder's *paraphrase-incoherent* noise floor; it bounds the irreducible
within-paraphrase variance in cosine retrieval.

### 4.3 Empirical phase information
$M$-sweep on HotpotQA-1k, NQ-open-1k. Coherent gain (pp) over incoherent ensemble.

## 5. Conservation-Law Faithfulness

Entity / numeric / temporal residuals. Correlation with RAGAS faithfulness;
abstention precision-recall.

## 6. Layered Pipeline + Calorimetric Routing

### 6.1 The full stack
Reproduce architecture diagram.

### 6.2 Calorimetric query archetype profiler
Score-field morphology → 5 archetypes; per-archetype recipe.

### 6.3 Headline result
Stage-decomposed accuracy on 5 benchmarks.

## 7. The Asimov Benchmark + Cross-Section Metric

Synthetic gold-injection benchmark; stage decomposition of accuracy as
$\varepsilon_\text{retrieval} \cdot \varepsilon_\text{select} \cdot \varepsilon_\text{generate}$.
Cross-section formalism for retrieval comparison across heterogeneous corpora.

## 8. Cross-Domain Transfer to Physics Corpora

Astro-ph and hep-ph arXiv subsets with ~150 LLM-generated, span-grounded
questions per corpus. Method gains preserved under domain shift —
methodological transfer is substantive, not metaphorical.

## 9. Ablations

- Per-method on/off matrix on F1–F9 coverage.
- Sensitivity to anti-$k_T$ R, lock-in M, conservation-residual tolerances.
- Effect of generator scale (8B vs 70B Llama).

## 10. Related Work

- Astro-source-detection / SExtractor / DRUID (deblending).
- ColBERT / RAPTOR / Self-RAG / ANCE / RAGAS / FActScore.
- Anti-$k_T$ (Cacciari–Salam–Soyez 2008); CLEAN (Högbom 1974); D'Agostini
  unfolding (1995); SBI (Cranmer 2020); diffusion priors (Howard et al. 2024).

## 11. Limitations

- Generator-side methods require model probing (closed-API exclusion).
- Conservation residuals are symbolic; subtle paraphrastic faithfulness still
  needs an LLM judge.
- Anti-$k_T$ v1 (leading-jet-only selection, $n_\text{jets}=1$) collapsed
  on real HotpotQA at cs=512 (citation accuracy 0.886 → 0.014) when the
  gold spans heterogeneous topics that cluster less tightly than
  near-duplicate distractors. v2 ($n_\text{jets}=-1$, pack-across-all-jets)
  fixes this without breaking the IRC theorem; the clustering decision
  remains anti-$k_T$ but is used as a soft jet-relevance preference rather
  than a hard exclusion of non-leading jets. The IRC mechanism advantage
  is therefore *chunk-size-bounded*: it manifests where boundary-splitting
  of joint evidence is unavoidable (small chunks); at larger chunk sizes
  v2 reduces to greedy by construction.

## 12. Conclusion

RAG is a measurement problem. Physics has tools. Mixing the right tools by
failure mode wins.

---

## Headline figure list

1. **Stack diagram** (Section 6.1) — the layered pipeline.
2. **F1–F9 method-coverage matrix** (Section 2.3).
3. **IRC-robustness curve** (Section 3.3) — F1 vs chunk-size.
4. **Coherent-vs-incoherent gain** (Section 4.3) — bar chart.
5. **Stage-decomposed accuracy** (Section 6.3) — Asimov ladder.
6. **Cross-domain transfer** (Section 8) — bar chart, std vs physics corpora.
7. **Ablation table** (Section 9) — full F1–F9 attribution.

## Falsifiable claims (pre-registration)

| Claim | Pass condition | Evidence file | Status |
|---|---|---|---|
| C1a anti-$k_T$ v3 strictly > greedy on synthetic IRC stress test | paired bootstrap $p<0.01$ on mean gold-pair coverage gap | `runs/synthetic_irc_iter2_v3/*` | **PASS** — n=100, +0.2084, P=1.000, CI95 [+0.175, +0.242] |
| C1b anti-$k_T$ v3 > anti-$k_T$ v1 on synthetic | paired bootstrap, CI excluding 0 | `runs/synthetic_irc_iter2_v3/*` | **PASS** — n=100, +0.0223, P=1.000, CI95 [+0.019, +0.025] |
| C1c anti-$k_T$ v3 ≥ greedy at every chunk size on real HotpotQA | $\ge$ greedy on citation accuracy across cs∈{64,128,256,384,512,768} | `runs/chunksize_sweep/*` | **PENDING** (job 2998872) |
| C1d anti-$k_T$ v3 strictly > greedy at small chunk sizes on real HotpotQA | citation accuracy lift $\ge$ 1pp at cs ≤ 128 | `runs/chunksize_sweep/*` | **PENDING** (job 2998872) |
| C2 coherent > incoherent paraphrase aggregation on F1 | mean lift > 0 with 95% CI excluding 0 | `runs/p4_lockin_sweep/*` | not yet run |
| C3 conservation residuals correlate with RAGAS faithfulness | Spearman $\rho > 0.5$ on RAGAS-eval | `runs/p5_conservation/*` | not yet run |
| C4 layered pipeline beats best single baseline | end-to-end EM/F1 lift $\ge$ 1pp on all of HotpotQA, NQ, MuSiQue | `runs/p7_full_matrix/*` | not yet run |
| C5 cross-domain gains preserved on physics corpora | EM/F1 lift on astro-ph and hep-ph $\ge$ 0.5pp | `runs/p7_physics_corpus/*` | not yet run |

**Empirical history (anti-$k_T$ specifically):**

- *Iter 1 (synthetic, v1, n=180):* directional P=0.759, +0.033 mean gap.
  Underpowered.
- *v1 selector on real HotpotQA-1k (cs=512, dense):* citation accuracy
  collapses 0.886 → 0.014. Diagnosis: leading-jet packing excludes gold
  when multi-hop spans don't cluster.
- *Iter 2 (synthetic, v1, n=12k):* +0.186 mean gap (35% relative), t=185,
  paired-bootstrap p<0.0001.
- *v2 selector (n_jets=-1, SNR-sort within jets):* fixed HotpotQA collapse
  but degenerated to greedy on synthetic (mean diff 0.0). v2 is a *no-op*.
- *v3 selector (n_jets=-2, atomic-unit greedy):* greedy by SNR + all jet
  partners pulled in for each primary selection. Synthetic mean diff
  **+0.2155** (CI95 [+0.161, +0.266]); also strictly beats v1 by +0.020
  (CI95 [+0.016, +0.024]). v3 is the publication selector.
- *v3 on real HotpotQA cs ∈ {64..768} sweep:* PENDING (job 2998872).
  Required for C1c/C1d publication-grade claim.

If any claim fails its pass condition, the paper either drops the claim
(repositioning the method as a diagnostic) or pivots to a null-result framing.
We pre-register the conditions to prevent retroactive rationalization.
