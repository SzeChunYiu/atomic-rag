# D03 — Staged Filtering Pipeline

## One-line claim
Restructure retrieval as L0 lexical/entity → L1 dense → L2 evidence-SNR
→ L3 atom-trigger → L4 cross-encoder rerank survivors → L5 final
selection, with each stage producing measurable signal-efficiency,
background-rejection, and latency curves.

## Bottleneck this targets
Current pipeline collapses Stages L0–L4 into a single dense + optional
rerank. The cost-quality frontier is opaque; we cannot tell which stage
contributes most efficiency vs which adds most cost.

## Five-question check
1. **Atom**: candidate chunks at each stage.
2. **Bottleneck**: undifferentiated retrieval/rerank cost; no
   tunable knobs for latency budgeting.
3. **Signal amplified**: per-stage signal/background ratio.
   **Noise suppressed**: stage-specific noise (lexical false-positives at
   L0, dense-only false positives at L1, etc.).
4. **Tradeoff**: more stages = more code complexity. Each stage adds
   latency unless it reduces downstream candidate count enough.
5. **Falsified if**: combined L0–L5 produces no Pareto improvement over
   plain dense+rerank in (F1, latency) space.

## Plan
- Phase 1: instrument each existing stage with signal_efficiency,
  background_rejection, p50/p95 latency.
- Phase 2: add L0 (BM25 lexical pre-filter) before dense.
- Phase 3: add L3 atom-trigger (entity-match check on top-K) between
  rerank and select.
- Phase 4: ablate stage count to find Pareto frontier.

## Status
Mostly orthogonal to D04. Could be implemented in parallel; deferred
to keep focus.

## Deliverables checklist
- [x] `summary.md`
- [ ] (rest deferred)
