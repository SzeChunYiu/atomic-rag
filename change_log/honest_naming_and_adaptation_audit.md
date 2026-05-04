# Honest naming + adaptation audit (2026-05-03)

User challenge: "We should not use the polished names — use the original.
But where is our novelty then? Did we optimize, or just copy?"

This document is the honest answer.

## Drop the physics labels in the paper

| Polished name | Real name (origin) | Drop in paper? |
|---|---|---|
| Anti-kT v4 jet selector | Score-gated agglomerative selection (kernel from Cacciari–Salam–Soyez 2008) | KEEP origin ref, drop "jet" framing |
| CLEAN-RAG | Iterative deconvolution (Högbom 1974) — direct port, FAILED | DROP — not in paper |
| Lock-in coherent sum | Phase-locked averaging — direct port, FAILED | DROP — not in paper |
| Path-Integral Retrieval (PIR) | Truncated random walk / Katz centrality (1953) — FAILED 3 variants | DROP — not in paper |
| Graph-Walk Retrieval (GWR) | Max-product / 1-hop expansion — null F1 | DROP — mention in negative-results appendix only |
| Evidence-SNR atoms | z-scored chunk weights | DROP — vocabulary inflation |
| Evidence Deblending (D04) | Sentence-level retrieval + claim-type tagging | RENAME to literal description |
| Sparse Reconstruction (D06) | Submodular set-cover (Krause–Golovin 2014) | RENAME to literal description |

## Where novelty actually lives

After stripping the labels, what's left:

1. **Bottleneck-decomposition methodology** — quantify loss per RAG stage; pick
   interventions at bottleneck. This is the methodological contribution.
2. **Score-gated α in agglomerative selection** — genuine algorithmic addition
   on top of anti-kT kernel. Predicted inverted-U, confirmed empirically.
3. **D04+D06 composition** — sentence-atoms × claim-typed facets × set-cover.
   Pieces are textbook; composition is ours.
4. **Co-tuning curves** (TO BE PRODUCED) — saturation plots over λ_type, top_k,
   token_budget, score_floor will turn "copies" into "adaptations" that are
   demonstrably tuned for the RAG context.

That's the novelty surface. Anything else in the paper is overclaim.

## What we did NOT optimize (the legitimate critique)

- λ_type ∈ {0, 0.05} only; no sweep
- top_k = 50 fixed
- token_budget = 1024 fixed
- score_floor = 0 (lets noise atoms in)
- score_bonus weight in submodular = 0.05 hand-set
- No joint co-tuning of selector × generator
- No learned components (tagger, facet extractor are regex)
- ANY-intent fallback is naïve (typed bonus simply disabled)

## Adaptation/optimization plan (after D04+D06 lands)

Sweep order (cheap → expensive, each gives a publishable curve):

1. λ_type ∈ {0, 0.025, 0.05, 0.1, 0.2, 0.5} on HotpotQA-1k
2. score_floor ∈ {0, 0.1, 0.2, 0.3}
3. score_bonus ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
4. token_budget ∈ {512, 1024, 1536, 2048} — saturation curve
5. top_k × token_budget 2D grid — interaction effect
6. Adaptive intent routing for ANY queries — query-specific facet expansion

These sweeps are what convert "we used submodular set-cover" into
"we tuned submodular set-cover for atom-level RAG selection".

## Paper rename plan for draft_v0.md

- "Sentence-level retrieval with claim-type facets" (was: D04 / Evidence Deblending)
- "Submodular set-cover selection over typed facets" (was: D06 / Sparse Reconstruction)
- "Score-gated agglomerative atom selection" (was: anti-kT v4)
- Title: *Bottleneck-driven RAG: an atomic decomposition framework
  for diagnosing and ablating loss in retrieval pipelines*

## Verdict

We have ~30% real novelty (methodology + α-gate + composition + forthcoming
sweeps) and ~70% standard tools. That is fine for Nature MI ONLY IF we
deliver the co-tuning curves. Without the sweeps, this is a methods-tour,
not a methods paper.
