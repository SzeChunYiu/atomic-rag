# The Layered Physics-Inspired RAG Pipeline

This is the architecture document. Every method in the project must declare
where in the stack it sits and which atomic failure mode (F1–F9) it owns.
A method without a stack position and a failure-mode owner is *not added*.

## The stack (top = generation, bottom = corpus)

```
                ┌─────────────────────────────────────────────┐
   Layer 8      │  Generation + abstention                    │
                │  Conservation-law residuals (HEP)           │  → F7, F9
                ├─────────────────────────────────────────────┤
   Layer 7      │  Context assembly                           │
                │  Adaptive-optics ordering (astronomy)       │  → F6
                ├─────────────────────────────────────────────┤
   Layer 6      │  Sparse evidence reconstruction             │
                │  Anti-kT jet clustering (HEP)               │  → F4, F8
                │  CLEAN-RAG residual loop (radio astronomy)  │  → F4, F2
                ├─────────────────────────────────────────────┤
   Layer 5      │  Selection / detection threshold            │
                │  Cherenkov hard cut (detector physics)      │  → F2
                │  FDR (BH) calibrated rejection (astronomy)  │  → F2, F5
                ├─────────────────────────────────────────────┤
   Layer 4      │  Background subtraction / SNR               │
                │  Aperture photometry (astronomy)            │  → F2
                │  Coronagraphy anchor mask (astronomy)       │  → F5
                │  Dark-matter-style background fit           │  → F2, F5
                ├─────────────────────────────────────────────┤
   Layer 3      │  Multi-channel score combination            │
                │  VLBI cross-correlation (radio astronomy)   │  → F1, F3
                │  Lock-in coherent paraphrase (lab physics)  │  → F1, F3
                ├─────────────────────────────────────────────┤
   Layer 2      │  Per-channel retrieval                      │
                │  Dense / BM25 / late-interaction / SPLADE   │  → baseline
                │  Hierarchical (RAPTOR-style)                │  → F4
                │  Optimal-transport (Wasserstein, 2024)      │  → F1
                ├─────────────────────────────────────────────┤
   Layer 1      │  Query reformulation                        │
                │  D'Agostini / diffusion-prior unfolding     │  → F3
                │  Anomaly-detection OOD gate (CATHODE 2024)  │  → routing
                ├─────────────────────────────────────────────┤
   Layer 0      │  Corpus ingestion                           │
                │  Claim-atom decomposer                      │  → all (substrate)
                │  TDA features of similarity graph           │  → routing
                │  Standard-candle calibration anchors        │  → calibration
                └─────────────────────────────────────────────┘
   Routing: Calorimetric query archetype profiler — chooses which Layer-2..6
   recipe to use per query (compact / plateau / bimodal / diffuse / noisy).
   Inference posterior (SBI 2024) — uncertainty-aware end-to-end selector.
```

## Mixing rule

Methods are mixed by **failure-mode coverage**, not by analogy density. The
target is *complete coverage* of F1–F9 with at least one method per failure,
ideally two for redundancy. The matrix below tracks coverage:

| Failure | Primary method (layer) | Secondary | Layer |
|---|---|---|---|
| F1 topical-but-wrong | Lock-in coherent paraphrase | Wasserstein OT, VLBI | 2/3 |
| F2 distractor swarm | Aperture-photometry SNR | Cherenkov, FDR | 4/5 |
| F3 query degeneracy | Diffusion unfolding | Lock-in, VLBI, OOD-gate | 1/3 |
| F4 split evidence | Anti-$k_T$ jet clustering | Hierarchical | 6 |
| F5 popular-but-empty | Coronagraphy | FDR, dark-bg | 4/5 |
| F6 lost-in-middle | Adaptive-optics ordering | Dead-time correction | 7 |
| F7 multi-source contradict | Conservation residuals | Coincidence-veto | 8 |
| F8 multi-hop bridge | Anti-$k_T$ via entity links | Heat-kernel diffusion | 6 |
| F9 generator drift | Conservation residuals | Asimov stage decomposition | 8 |

**A method without an F-row is not in the pipeline.**

## End-to-end objective (P6 unification)

We will write a single Bayesian objective whose minimization recovers the
stack as its variational solution: layers are terms in the posterior
$p(\text{answer}, E \mid q)$, calibrated against the Asimov benchmark, with
the calorimetric routing as a learned mixture-of-experts. P6 is the place
where the layered pipeline becomes the *paper* — a derivation, not a list.

## What this means for this project

- Implementing a method = adding a row to the matrix.
- Beating baselines = closing F-rows with theorems + ablations.
- The paper = the stack + its derivation + the failure-coverage table.
