# D04 — Evidence Deblending

## One-line claim
Splitting 384-token chunks into structured evidence atoms with typed
slots (entity, number, date, claim, direction, negation) lets selection
score per-atom rather than per-chunk, recovering bridge-evidence atoms
that chunk-level SNR currently buries.

## Bottleneck this targets
Documented atomic chain on HotpotQA-1k cs=384:
- Retrieval recovers bridge chunks at +6pp (graph propagation)
- Atom-SNR drops them because cos(query, chunk) is low for the bridge
  hop (the chunk that connects, but doesn't directly answer)
- Selection budget skips low-SNR atoms

D04 attacks the second arrow: replace cos-based chunk-SNR with
structured atom-level matching that scores each atom by its claim type
and entity-relation match to the query, not by global cosine.

## Five-question check (per directions doc)

1. **What atomic object does this method operate on?**
   The evidence atom: `(claim_text, claim_type, entities, source_chunk_id,
   span_start, span_end, confidence)`.

2. **What bottleneck or failure mode does it target?**
   The atom-SNR layer currently scores chunks by `cos(q, chunk_emb)`.
   Bridge chunks have low cos but contain the answer entity. By
   extracting typed atoms, a chunk that doesn't match the query overall
   can still expose its answer-bearing atom.

3. **What signal is amplified, and what noise is suppressed?**
   Signal: structured claim types matching query intent (e.g., "WHO ?"
   matches PERSON entities; "WHEN ?" matches DATE atoms).
   Noise: distractor sentences within an otherwise-relevant chunk.

4. **What tradeoff is introduced?**
   - Cost: NER/parse pass over corpus (one-time).
   - Recall risk: if the deblender misses an atom, that atom can never
     be selected.
   - Brittleness: domain-specific extraction patterns.

5. **What exact result would falsify the method?**
   - F1 on HotpotQA-1k <= F1 of greedy+rerank+CoT (currently 0.633).
   - all_atom_in_sel does NOT increase relative to all_chunk_in_sel.
   - Atom extraction adds >50ms/query latency without F1 gain.

## Plan

**Phase 1 (this iteration):** Minimal viable deblender that splits
chunks at sentence boundaries and tags each sentence with claim-type
heuristics (WHO/WHEN/WHERE/WHAT/HOW). Replace chunk-level retrieval
with atom-level retrieval. Score atoms by `cos(q_emb, atom_emb)` PLUS
a typed bonus when claim_type matches query intent.

**Phase 2 (defer):** Full slot extraction with spaCy NER, relation
extraction patterns, contradiction detection.

## Deliverables checklist (per directions doc)
- [x] `summary.md`
- [ ] `prior_work.md`
- [ ] `novelty.md`
- [ ] `algorithm.md`
- [ ] `bottleneck_analysis.md`
- [ ] `benchmark.md`
- [ ] `failure_cases.md`
- [ ] `implementation_spec.md`
- [ ] `decision.md`
