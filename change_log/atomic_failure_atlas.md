# Atomic failure atlas

The mechanism map every method must reference. Each atomic failure mode is
defined by a **single, falsifiable mechanism**, an **observable signature**
(measurable from artifacts), and a **method assignment** (which physics-inspired
method addresses it primarily).

## The taxonomy

### F1 — topical-but-wrong
**Mechanism.** Encoder mixes topical and relational dimensions; cosine cannot
disentangle "about X" from "asserts X" or "contradicts X."
**Observable.** ScoreShape: high `peak_score`, but cited gold absent at top-k.
Score-rank vs gold-rank correlation collapses on relation-typed queries.
**Method.** Matched filter with relation phase (Phase 4 lock-in variant).

### F2 — distractor swarm
**Mechanism.** Gold sits inside a dense cluster of similar non-gold; absolute
score is uninformative.
**Observable.** ScoreShape archetype = `plateau` or `bimodal` with low
`second_peak_gap`; many candidates within ε of peak.
**Method.** Local-background SNR (already in `detection/snr.py`) + Fano-factor
entity reweighting (P3+).

### F3 — query degeneracy
**Mechanism.** Multiple latent needs map to similar query embeddings; the
embedding $\hat q$ is a smeared image of $\theta^*$.
**Observable.** Lock-in coherent paraphrase score < incoherent ensemble (in
the limit, the difference *is* the noise budget). Spread of score across
paraphrases is large.
**Method.** Lock-in coherent paraphrase (P4) and D'Agostini unfolding (P6).

### F4 — split evidence
**Mechanism.** A single claim spans a chunk boundary; both halves underscore;
selection cannot recombine them.
**Observable.** Two consecutive chunks of the same doc, both rank ~middle,
combine into a high-relevance unit only after merging.
**Method.** Anti-$k_T$ IRC-safe evidence-jet clustering (P3) — by construction.

### F5 — popular-but-empty
**Mechanism.** A chunk has high prior retrievability across all queries
(common phrasing, frequent vocabulary) without information for any specific
query.
**Observable.** A chunk consistently appears in candidate sets across many
unrelated queries; conditional retrievability ≫ marginal retrievability.
**Method.** Dark-matter-style background subtraction (P3+) and Fano-factor
shot-noise floor.

### F6 — lost-in-middle
**Mechanism.** LLM positional decay; relevant evidence in middle of context
gets attended weakly.
**Observable.** Same selected set, two orderings → very different EM/F1.
**Method.** Adaptive-optics context pre-distortion (P5+).

### F7 — multi-source contradiction
**Mechanism.** Two correct-looking chunks disagree; the LLM averages.
**Observable.** Generated answer text contains hedges or both claims;
faithfulness judges flag conflict.
**Method.** Coincidence-veto detection + conservation-law residuals (P5).

### F8 — multi-hop bridge
**Mechanism.** Gold chunk holds a bridge entity, not a literal answer; needs
a second hop.
**Observable.** Recall@k high, EM low; gold appears but generator can't
synthesize because the next-hop chunk is not in context.
**Method.** Causal-cone diffusion / heat-kernel multi-hop (P3+).

### F9 — generator drift
**Mechanism.** Right context, wrong synthesis — generator hallucination or
omission despite gold-bearing context.
**Observable.** Asimov benchmark gap: with gold inserted at known position,
EM/F1 still below 1.0.
**Method.** Conservation-law verification (P5) + abstention.

## Method-to-failure assignment matrix

| Method (planned) | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 |
|---|---|---|---|---|---|---|---|---|---|
| Local-background SNR (already P0) | – | ✔ | – | – | ◯ | – | – | – | – |
| Anti-$k_T$ jet clustering (P3) | – | ◯ | – | ✔ | – | – | – | ◯ | – |
| Lock-in coherent paraphrase (P4) | ✔ | – | ✔ | – | – | – | – | – | – |
| Dark-matter background (P3+) | – | ✔ | – | – | ✔ | – | – | – | – |
| Heat-kernel multi-hop (P3+) | – | – | – | – | – | – | – | ✔ | – |
| Conservation residuals (P5) | – | – | – | – | – | – | ✔ | – | ✔ |
| Adaptive-optics order (P5+) | – | – | – | – | – | ✔ | – | – | – |
| Calorimetric routing (P5+) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

✔ = primary fix, ◯ = secondary contribution.

The matrix is the contract: when a method enters the pipeline, its row must be
empirically validated against the corresponding failure modes — otherwise it
is removed.

## Per-archetype empirical mapping (filled by P1 profiler runs)

The calorimetric profiler (`pipeline/profile_run.py`) produces an archetype
histogram + mean recall-per-archetype. Once we run the dense-rerank baseline
on HotpotQA-1k and NQ-open-1k, we will fill the table below with empirically
observed associations:

| Archetype | Hypothesized dominant failure | Empirical recall@10 (TBD) |
|---|---|---|
| compact | F1 / F8 | — |
| plateau | F2 / F5 | — |
| bimodal | F7 | — |
| diffuse | F3 / F4 | — |
| noisy | F3 / F9 | — |

Until those numbers exist, no method should claim it "addresses" an archetype.
