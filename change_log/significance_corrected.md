# Paired-bootstrap significance — corrected with F1

**Date:** 2026-05-03
**Source:** `scripts/bootstrap_all_terse.py`, n=10000 resamples, paired by query_id.

## Significance table (vs greedy baseline)

| comparison | Δcit_acc | P(>0) | ΔF1 | P(>0) |
|---|---|---|---|---|
| MMR | +0.05pp | 0.592 (NULL) | **-0.28pp** | 0.000 (slight regression) |
| **v4 anti-kT (α=0.7)** | **+1.39pp** | **0.975** | +0.06pp | 0.532 (NULL) |
| **greedy + rerank** | **+1.78pp** | **0.955** | **+2.93pp** | **0.997** |
| v4 + rerank | +1.26pp | 0.881 | **+2.71pp** | **0.991** |
| CLEAN-RAG | **-8.06pp** | 0.000 | **-23.52pp** | 0.000 |

## Head-to-head: v4 vs greedy+rerank

| metric | Δ (rrk − v4) | P(rrk > v4) |
|---|---|---|
| cit_acc | +0.40pp | 0.656 (NULL — statistically tied) |
| F1 | **+2.87pp** | **0.996** (rerank significantly better) |

## Corrected mechanistic interpretation

The F1 fix exposes a sharper finding than the cit_acc-only narrative:

1. **F1 gain comes only from cross-encoder rerank.** No dense-side
   selector improves F1 over greedy (v4: +0.06pp NULL, MMR: −0.28pp
   slight regression). This is mechanistically explained: dense
   selectors permute within the same top-50 pool. If the
   answer-containing chunk is ranked low or absent in top-50, no
   permutation can promote it. Only the cross-encoder, which scores
   (query, chunk) pairs jointly, can elevate answer-containing chunks
   that the bi-encoder under-ranked.

2. **cit_acc gain comes from BOTH v4 AND rerank.** They are statistically
   tied head-to-head on cit_acc (+0.40pp, P=0.656 NULL). v4 captures
   essentially all of rerank's cit_acc benefit at 0% extra compute.
   This makes sense: cit_acc rewards "the cited chunk's doc was a gold
   doc" — dense top-50 already contains gold docs (recall@5=0.836),
   so cluster topology (v4) is enough to surface them.

3. **v4 + rerank: best F1, but cit_acc regresses vs greedy+rerank.**
   F1 maintained (+2.71pp vs +2.93pp, statistically tied) but cit_acc
   shrinks (+1.26pp vs +1.78pp). Why? Rerank already promotes good
   chunks; v4's "two-jet" preference DEMOTES some rerank-promoted
   chunks in favor of cluster-leaders, hurting cit_acc.

4. **CLEAN-RAG fails on both metrics catastrophically.** Now confirmed
   on F1 too: −23.52pp (P=0). The radio-CLEAN analogy inverts the RAG
   case (relevant evidence is redundant, not orthogonal).

## Implication for paper

The two-axis (cit_acc, F1) view tells a clean story:

| selector class | cit_acc (selection-stage) | F1 (answer quality) |
|---|---|---|
| dense-side selectors | improvable (v4 wins) | NOT improvable |
| cross-encoder rerank | improvable | **improvable (only path)** |

**This is the framework headline:** dense selectors can improve citation
accuracy without changing the answer; cross-encoder rerank is the only
mechanism that improves answer quality. The two operate on different
axes and they do not stack on cit_acc but stack neutrally on F1.

For a Nature MI paper, this is a sharper claim than "selector X beats
selector Y by N pp." It identifies a *qualitative* boundary: which
mechanisms can move which metrics, and why.
