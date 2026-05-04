# Atomic corrections — what saturation actually means and why CoT works

**Date:** 2026-05-03

## Two corrections to earlier claims

### Correction 1: "selection is saturated" was the wrong frame

Earlier I reported `gold_in_sel = 99.7%` and called selection saturated.
That's "ANY gold doc in context" — which is necessary but not sufficient
for multi-hop bridge queries (which need TWO gold docs to chain through).

Re-measured with `all_gold_in_sel` (ALL gold docs in context):

| selector | any_gold | **all_gold** | F1 (when all gold present) | F1 (when partial) |
|---|---|---|---|---|
| greedy | 0.996 | **0.856** | 0.614 | 0.194 |
| greedy+rerank | 0.997 | **0.893** | 0.632 | 0.171 |

Two atomic findings:
1. **Selection is NOT saturated.** 14.4% of queries (greedy) and 10.7%
   (greedy+rerank) lack at least one gold doc in context.
2. **F1 collapses 3.2× on these queries** (0.61 → 0.19). The entire F1
   advantage of rerank over greedy is captured by the +3.7pp improvement
   in `all_gold_in_sel` — same all-gold F1, fewer partial-context queries.

### Tracing the missing-gold queries

Of 107 queries with all_gold missing on greedy+rerank:
- **106 (99%): the missing gold doc was never in top-50 dense retrieval.**
- 1: lost to token budget.
- 0: dropped by rerank.

**The bottleneck is dense retrieval recall, not selection or rerank.**
Bi-encoder cosine similarity misses the second gold doc when it's about
an entity not mentioned in the query (the answer entity in a bridge
chain).

This is **exactly the percolation problem PIR was designed for** — the
2nd gold doc is reachable via a multi-hop path through the
chunk-similarity graph (1st gold doc shares the bridge entity with the
query) but not by direct query-chunk cosine.

### Correction 2: CoT delivers +5pp F1, not +20pp

Earlier I claimed CoT lifted F1 from 0.40 → 0.60. The 0.40 came from the
run's metrics.json which uses one normalizer; the 0.60 came from my
post-processing script which uses a different normalizer (strips `[E_i]`
markers from the answer text, which inflates token-overlap with gold).

Apples-to-apples with the same normalizer:

| config | F1 | EM |
|---|---|---|
| terse | 0.553 | 0.436 |
| **CoT** | **0.604** | **0.471** |
| Δ | **+5.18pp** | **+3.5pp** |

Atomic decomposition of CoT effect (n=1000):

| bucket | count | % |
|---|---|---|
| both correct (EM) | 389 | 38.9% |
| **CoT wins (CoT EM, terse wrong)** | **82** | **8.2%** |
| **CoT loses (terse EM, CoT wrong)** | **47** | **4.7%** |
| both wrong, both F1=0 | 253 | 25.3% |
| both partial, CoT better | 50 | 5.0% |
| both partial, terse better | 29 | 2.9% |
| both partial, similar | 150 | 15.0% |

**Net: CoT helps 8.2%, hurts 4.7%, neutral 87.1%.**

The +5pp F1 / +3.5pp EM is real but modest — not a breakthrough. The
mechanism is that CoT lets the model verify intermediate entities
explicitly, which fixes some bridge-query failures. But CoT also INTRODUCES
new failure modes on 4.7% of queries — likely cases where the explicit
reasoning step misleads the model.

Average answer lengths are similar (2.3 terse vs 2.2 CoT words), so the
F1 gain isn't a length artifact.

## What this means for the paper

The atomic decomposition pinpoints the actual bottlenecks:

| stage | bottleneck | size | mechanism that addresses it |
|---|---|---|---|
| 1-3 chunk + embed | embedding manifold loses bridge structure | causes missing-gold | better embedder OR multi-hop graph |
| 4 dense retrieval top-50 | **misses 2nd gold doc on 10-14% of bridge queries** | F1 0.17 on these | **PIR — physics-motivated** |
| 5 rerank | doesn't add coverage; reorders within top-50 | recovers 4pp all_gold | works for already-retrieved chunks |
| 6-7 atom + select | minimal | <1% | not a bottleneck |
| 9 generation | extracts wrong answer on 8% of bridge queries even with all gold present | F1 0.61 ceiling | CoT addresses; +5pp F1 |

**Two attackable bottlenecks remain:**
1. Dense retrieval missing the chained gold doc (10.6% of queries) →
   PIR is the right physics method.
2. Generation extracting wrong answer despite all-gold present (CoT
   fixes 8.2% but introduces 4.7% new failures, net +5pp F1) →
   need to characterize WHY CoT fails on the 4.7% before proposing fix.

## What "saturated" actually means

A stage is *saturated* iff the metric it controls cannot move under any
mechanism applied to that stage.

- Selection IS saturated for `gold_in_sel` (any-gold), but NOT for
  `all_gold_in_sel` (the bottleneck for multi-hop).
- Citation IS saturated GIVEN the answer (citations track answer text),
  but NOT for cit_acc — because cit_acc is gated by answer correctness.
- Retrieval is NOT saturated for bridge queries — 10.6% of gold docs
  miss top-50.
- Generation is NOT saturated — CoT showed +5pp F1 headroom; further
  mechanisms (self-consistency, decomposition) likely add more.

This is the corrected mental model for the paper.
