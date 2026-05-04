# D06 — Sparse Evidence Reconstruction

## One-line claim
Replace greedy budget-fill with a submodular set-cover objective:
maximize coverage of the query's needed evidence dimensions while
minimizing redundancy and contradictions, subject to token budget.

## Bottleneck this targets
After D04 lands typed atoms, the selector must choose among them.
Greedy by SNR is suboptimal because it doesn't reward COVERAGE of
distinct query facets. Submodular set-cover with a coverage utility
provably approximates the optimal selection within (1 − 1/e) ≈ 63%.

## Five-question check
1. **Atom**: a typed evidence atom (post-D04).
2. **Bottleneck**: greedy selection picks redundant atoms when
   multiple high-SNR atoms cover the same query facet, leaving
   low-SNR but uncovered facets out.
3. **Signal amplified**: coverage of distinct query facets.
   **Noise suppressed**: redundant within-facet atoms.
4. **Tradeoff**: needs facet decomposition of the query (cheap heuristic
   or LLM-light) and facet membership tagging on atoms.
5. **Falsified if**: submodular selection F1 ≤ greedy-by-SNR F1 across
   all tested datasets.

## Plan
- Phase 1: define facets as the typed-claim-types from D04.
- Phase 2: greedy submodular maximization over typed atoms.
- Phase 3: ablate facet decomposition granularity (5 types vs 10 vs
  query-LLM-extracted).

## Status
Deferred until D04 atoms are validated. Implementation will be ≤ 100
lines once D04 is in place.

## Deliverables checklist
- [x] `summary.md`
- [ ] (rest deferred)
