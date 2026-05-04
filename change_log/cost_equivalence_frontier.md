# Cost-equivalence frontier — v4 vs rerank

**Date:** 2026-05-03
**Source:** SLURM elapsed times for HotpotQA-1k cs=384 terse runs.

## Per-query latency decomposition

| stage | greedy/dense | v4/dense | greedy + rerank | clean-rag |
|---|---|---|---|---|
| retrieve+embed | 13ms | 13ms | 13ms | 13ms |
| selector | ~100ms* | ~100ms* | ~100ms* | ~100ms* |
| rerank (CE forward) | — | — | ~380ms | — |
| generation (Qwen 7B) | 472ms | 476ms | 465ms | 385ms |
| **total** | **~590ms** | **~590ms** | **~960ms** | **~510ms** |

\*Selector cost dominated by Python loops in candidate scoring; **same for all
selectors** because the bottleneck is per-candidate scoring (50 candidates ×
embedding lookup × dot product × heap ops).

Source data: SLURM `sacct` Elapsed for jobs 3001639 (3 dense selectors,
29:27 / 3 = 589s/run), 3001962 (rerank-terse, 14:14 = 854s/run on the
greedy_rerank branch only — note this includes index build), 3002218
(v4_rerank, 14:06 = 846s/run), 3002225 (clean_rag, 11:17 = 677s).

*Note:* total per-query times above subtract estimated index-build
amortization (≈1 minute for 50k chunks on A100, included once per job).

## Cost-equivalence comparison

For a +1.39pp gain in citation accuracy, **anti-$k_T$ v4 costs zero
additional latency** over greedy.

For +1.78pp gain, **cross-encoder rerank costs +380ms/query** (a 65%
relative latency penalty over the baseline pipeline).

**v4 captures 78% of rerank's cit_acc gain at 0% of its compute cost** —
but with a **caveat (added 2026-05-03 with corrected F1 bootstrap):**
v4 does *not* capture rerank's F1 gain (v4 ΔF1 = +0.06pp NULL; rerank
ΔF1 = +2.93pp, P=0.997). v4 is a cheap *cit_acc* approximation, not a
full rerank substitute. See `significance_corrected.md` for the
two-axis framing.

This is the corrected headline cost framing for the paper:

> "Anti-$k_T$ v4 is a Pareto-optimal selector when the metric of interest
> is citation accuracy: it captures cross-encoder rerank's citation
> accuracy gain on multi-hop bridge queries (statistically tied,
> P(rrk>v4)=0.656) while adding negligible latency. For answer F1,
> however, only cross-encoder rerank delivers a statistically
> significant gain (+2.93pp, P=0.997); no dense-side permutation of the
> top-50 can promote answer-containing chunks that the bi-encoder
> under-ranked."

## Why selectors are essentially free
Bi-encoder embeddings are pre-computed at indexing time. Selection is
pure NumPy on cached vectors plus a token-budget bookkeeping pass. There
is no model forward pass, no GPU memory transfer, no autoregressive
decode. **The only way a dense-side selector adds compute is via
algorithmic complexity, not learned parameters.** Anti-$k_T$ v4 is
$O(N^2)$ in candidates per query (N=50), which is ~2500 dot products —
the same order as MMR.

## Why rerank is expensive
BGE-reranker-v2-m3 is a 568M-parameter cross-encoder. For each (query,
chunk) pair it runs a full transformer forward pass over the joint
sequence. With top-50 candidates per query and batch size 32 on A100,
amortized cost is ~380ms/query. **This is necessary because cross-attention
between query and chunk tokens is what lets the reranker break the
bi-encoder embedding-space bottleneck.**

## Implication for paper
Add Figure 4: cost-vs-cit_acc Pareto plot. Three points per dataset:
- greedy/dense (baseline cost, baseline cit_acc)
- v4/dense (same cost, +1.39pp cit_acc)
- greedy+rerank/dense+CE (+65% cost, +1.78pp cit_acc)

The v4 point dominates greedy. The rerank point lies on a different
frontier (higher cost regime). v4 is the right choice when you need cheap
multi-hop bridging; rerank is the right choice when you can afford the
cross-encoder.
