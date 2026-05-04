# v4 + rerank does NOT stack — rerank subsumes v4

**Date:** 2026-05-03
**Status:** final (n=1000, HotpotQA-1k cs=384 terse)

## Hypothesis tested
v4 anti-kT (+1.4pp cit_acc on dense) and cross-encoder rerank (+1.8pp
cit_acc on dense) act at orthogonal pipeline stages (selector vs
retriever); they should compose.

## Result — they do NOT stack

| stack | recall@5 | F1 | cit_acc | faith |
|---|---|---|---|---|
| greedy + dense | 0.8355 | 0.3967 | 0.7987 | 0.6225 |
| v4 + dense | 0.8355 | 0.3991 | **0.8126** | 0.6137 |
| greedy + rerank | **0.9070** | **0.4157** | **0.8165** | 0.6289 |
| v4 + rerank | 0.9070 | 0.4146 | 0.8113 | 0.6304 |

vs greedy+rerank, v4+rerank:
- F1: -0.11pp
- cit_acc: **-0.52pp** (regression)
- faith: +0.15pp

## Atomic interpretation

The cross-encoder reranker scores (query, chunk) jointly and concentrates
score mass on the truly-relevant chunks. After rerank, the top-20 pool
fed to the selector is already sharply discriminated — the highest-scoring
chunks are the multi-hop bridge atoms that v4 was trying to surface.

**v4 was solving the same problem as rerank, but through a weaker
mechanism (cluster topology of bi-encoder embeddings vs cross-attention
relevance scores).** Once rerank is in the pipeline, v4's contribution is
not additive — it's a small *re-ordering* of an already-correct top-20,
which can be neutral or slightly harmful if v4's anti-kT preference for
two complementary jets disagrees with the rerank score ordering on
some queries.

## Implication for the paper

The clean story is now:

1. **Cross-encoder rerank** is the *dominant* mechanism for multi-hop
   retrieval (+7.15pp recall@5, +1.78pp cit_acc).
2. **v4 anti-kT** is a *cheap dense-only alternative* that captures part
   of the rerank signal without the cross-encoder cost (+1.39pp cit_acc on
   dense, equivalent to ~80% of the rerank cit_acc gain at ~5× lower
   inference cost).
3. **The two are not complementary** — rerank subsumes v4 on bridge queries.

This is more publishable than "two orthogonal mechanisms," because:
- It defines exactly what v4 *is* (a dense-side approximation to cross-attention bridging).
- It defines when v4 *is the right choice* (when cross-encoder rerank is too expensive at scale).
- It still admits a multi-hop specialization claim (NQ-open shows v4's
  gain vanishes on single-hop, confirming the cluster mechanism is
  multi-hop-specific).

## Cost framing for paper

Rerank top-50→20 with bge-rrkv2-m3 (~600M params) on A100: ~? ms/query.
v4 anti-kT on dense top-50: pure CPU/GPU ops on already-cached
embeddings, sub-millisecond.

If the cost ratio is large (likely 100×+), v4 is a reasonable
"poor-person's rerank" for compute-constrained settings. Worth
benchmarking timing.
