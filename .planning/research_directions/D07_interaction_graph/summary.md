# D07 — Interaction Graph Over Evidence Atoms

## One-line claim
Build a typed-edge graph over evidence atoms (supports, contradicts,
causes, before, after, depends_on, same_as) and select a stable,
low-contradiction subgraph that covers the query's claim-type needs.

## Bottleneck this targets
Current graph-walk retrieval (GWR) uses untyped cosine edges and
recovers chunks that mention the same entity, regardless of the
relation. Many false-positive bridge chunks are "topically related but
contradictory" — they mention the same entity in a different timeframe
or with a different attribute.

## Five-question check
1. **Atom**: typed evidence atom (post-D04). Edge: a relation between
   two atoms with a confidence score.
2. **Bottleneck**: untyped graph propagation pulls in topically related
   distractors. Need typed edges to discriminate "X-was-director-of-Y"
   from "X-was-actor-in-Y".
3. **Signal amplified**: consistent relation chains (A supports B
   supports C). **Noise suppressed**: contradictions and irrelevant
   topical co-occurrences.
4. **Tradeoff**: needs a relation extractor (LLM-light or pattern-based).
   Significant offline preprocessing.
5. **Falsified if**: typed-edge propagation gives no F1 gain over
   untyped GWR (when D04 atoms are used).

## Plan
- Phase 1 (deferred): pattern-based relation extraction
  (subject-verb-object triples) on each chunk.
- Phase 2 (deferred): LLM-light relation tagging via Qwen 1.5B in batch
  mode.
- Phase 3 (deferred): typed-edge graph propagation.

## Status
Most ambitious of the four directions. Defer until D04 + D06 are
validated and deliver measurable gain. Without those, D07's
prerequisites are not in place.

## Deliverables checklist
- [x] `summary.md`
- [ ] (rest deferred)
