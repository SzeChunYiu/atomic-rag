# Paper structure v2 (2026-05-03)

User reframe: novelty = physics/astronomy methods not previously studied in
RAG. But each must be ADAPTED, not ported.

## Structural backbone

For every method we discuss, the paper must answer:

1. **Origin** — exact citation, what problem the method solves in physics/astro.
2. **Vanilla failure on RAG** — what assumption of the source domain breaks.
   This must be empirically demonstrated, not asserted.
3. **Adaptation** — concrete algorithmic change we made.
4. **Ablation** — vanilla port vs our adapted version on the same data.
   Adaptation must beat vanilla by a measured Δ.

If any row is missing, the method goes in the negative-results appendix.

## Method-by-method status

### KEEP (paper-ready or path to paper-ready)

**Score-gated agglomerative atom selection** (kernel: anti-kT, Cacciari–
Salam–Soyez 2008)
- Origin: jet clustering in collider physics
- Vanilla failure: pure clustering wastes token budget on intra-cluster
  redundancy with no quality gating
- Adaptation: score-gated partner pull-in (α parameter)
- Ablation: α ∈ {0, 0.5, 1, 2, 5}, inverted-U at α=1, both endpoints fail
  in predicted physics-meaningful way (α=0 → over-cluster; α large → no
  partner pull-in, equivalent to greedy)
- **Status: paper-ready**

**Submodular set-cover with claim-typed facets** (kernel: Krause-Golovin
sensor-placement-style submodular maximization)
- Origin: OR/ML; submodular budget-constrained selection
- Vanilla failure (to be measured): without claim-type structure, set-cover
  treats sentences as bag-of-tokens, missing the type-of-evidence axis that
  HotpotQA bridges actually need
- Adaptation: facets = (claim_type × entity); coverage requires type-match;
  score_bonus weighting tuned for RAG
- Ablation: SWEEPS QUEUED on LUNARC (λ_type, token_budget, top_k,
  score_floor, score_bonus)
- **Status: pending sweeps; paper-ready if curves are non-trivial**

**Sentence-level retrieval with claim-type tagging** (D04 — RAG-native, no
clean physics origin)
- Origin: NLP (sentence-BERT); claim-type tagging from QA literature
- Justification: makes the typed-facet machinery above usable
- **Status: enabling infrastructure, not a headline contribution**

### NEGATIVE RESULTS (instructive failures — appendix)

**Iterative deconvolution** (Högbom 1974 CLEAN)
- Origin: radio interferometry, sparse sky deconvolution
- Vanilla failure: assumes residual is a NEW source. In RAG, residual is the
  *unexplained query intent*, not a new evidence source. Subtracting matched
  components inverts the search direction.
- Adaptation: not attempted (would require redefining "residual" as
  uncovered-facet vector, not embedding leftover)
- **Status: negative result, paper as cautionary transfer**

**Phase-locked coherent averaging** (lock-in)
- Origin: signal processing under noise
- Vanilla failure: paraphrase queries are not phase-coherent; coherent sum
  averages out, doesn't accumulate
- **Status: negative result**

### NEEDS RESCUE OR DROP

**Truncated random walk / Katz centrality** (was: "PIR")
- Origin: graph network analysis (Katz 1953)
- Empirical failure: top-50 subgraph too dense, propagation overwhelms
  direct query signal
- Adaptation idea: prune subgraph to k-NN-of-query before propagation, OR
  use direct-walk weight that decays with subgraph density
- **Decision: implement one targeted adaptation; if still null, drop**

**1-hop max-product expansion** (was: "GWR")
- Origin: belief propagation, Viterbi
- Empirical failure: +6pp coverage at retrieval but null F1 (gain absorbed
  by atom-selector SNR floor)
- Adaptation idea: co-tune retrieval expansion with selector facets — add a
  "graph-distance facet" to set-cover so propagated atoms get explicit
  coverage credit
- **Decision: try after D04+D06 sweeps land; clean co-tuning experiment**

## Paper-shape implications

- **Title**: *Physics-inspired methods for atomic RAG: what transfers, what
  doesn't, and how to adapt what does*
- **Section 1**: bottleneck-decomposition methodology
- **Section 2**: kept methods (anti-kT-gated selection, set-cover with
  typed facets), each with the 4-bar structure above
- **Section 3**: negative results (CLEAN, lock-in) as cautionary transfers
  — these are valuable contributions if framed correctly
- **Section 4**: co-tuning curves (the sweep results)
- **Section 5**: best stack vs published baselines

This is a Nature MI-shaped paper because:
1. Cross-disciplinary methodology (physics → ML)
2. Honest failure analysis (negative results published)
3. Empirical adaptation curves (saturation, inverted-U)
4. Beats strong RAG baselines (need to confirm post-sweep)
