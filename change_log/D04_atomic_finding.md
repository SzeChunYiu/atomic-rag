# D04 atom-level retrieval — atomic finding (2026-05-03)

## Headline

Atom-level dense retrieval ALONE is a regression vs chunk-level retrieval
on HotpotQA-1k. Typed-bonus retrieval makes it slightly worse.

| Variant | F1 | EM | cit_acc |
|---|---|---|---|
| Chunk-greedy + Qwen 7B (baseline) | ~0.58 | ~0.45 | ~0.69 |
| D04-bare  (λ_type=0)              | 0.482 | 0.380 | 0.700 |
| D04-typed (λ_type=0.05)           | 0.476 | 0.371 | 0.689 |

Δ vs chunk baseline: **-10pp F1**. Δ typed vs bare: **-0.6pp F1**.

## Atom catalog stats

- 65,046 atoms from 20,017 chunks (3.25 atoms/chunk)
- WHO 33%, WHEN 27%, WHAT_OBJ 28%, WHERE 9%, WHAT_NUM 2%, ANY 0%

ANY=0 is the smoking gun: the tagger labels every sentence with a type
(via regex fallback when no rule fires), so typed retrieval has no
type-agnostic fallback pool. λ=0.05 then over-routes to one type and
kills recall on entities that live in another type.

## Atomic decomposition of the regression

| Stage | Failure mechanism |
|---|---|
| Embedding | BGE-M3 trained on chunk-length text; sentence embeddings are noisier |
| Retrieval | Top-50 atoms cluster in 5–10 docs; bridge docs missed |
| Selection | Greedy budget-fill doesn't enforce two-doc coverage |
| Generation | 50 isolated sentences ≠ surrounding context for disambiguation |
| Typed bonus | All atoms typed → bonus concentrates one type, kills cross-type recall |

The dominant damage: **bridge-doc coverage**. HotpotQA bridges need atoms
from both source docs; atom retrieval over-concentrates by document.

## Wait-state before next experiment

D04+D06 is running (3004138). Submodular set-cover with claim-typed
facets is *exactly* the fix for over-concentration — facets enforce
diverse coverage. If D06 closes the -10pp gap, we don't need any further
adaptation. If D06 only partially closes it, the next adaptation is ATCP
below. Don't pre-implement; let the chain settle.

## Next adaptation if D06 doesn't fix it: ATCP (atom-targeted, chunk-presented)

Decouple retrieval grain from generation grain:
- Retrieval: atom-level (precision targeting)
- Generation: each selected atom expanded back to parent chunk (context)

Physics analogy: tracker (high res, position) + calorimeter (lower res,
energy) — the high-res hit picks the seed; the lower-res context
identifies the particle. RAG analog: atom score picks the SEED, chunk
context lets the generator disambiguate.

This is genuinely not in published RAG work I know — most atom-level RAG
approaches generate on atoms. **Ours: retrieve on atoms, generate on
chunks.** Single-line code change in the bench script (replace
`s["text"]` with `chunk_text[s["chunk_id"]]` when assembling evidence).

## Also: tagger fix

If we keep typed retrieval, do one of:
1. Confidence-thresholded tagging — below threshold → ANY
2. Soft type distribution rather than hard label
3. Per-query λ_type gating — only apply bonus if query intent is high-confidence

(2) is correct; (1) is cheap; (3) is independent of the tagger.

## Decision pending

Hold ATCP and tagger-fix until D04+D06 lands. The honest research move
is to let the next layer's experiment finish before adding another
layer.
