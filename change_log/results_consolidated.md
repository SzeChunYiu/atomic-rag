# Consolidated results — Astro-CS-RAG

**Last updated:** 2026-05-03
**Stack:** BGE-M3 retrieval (top-50), Qwen 2.5-7B-Instruct generator (terse prompt), citation_accuracy = fraction of cited chunks whose doc_id ∈ gold.

## Headline finding

> **Cross-encoder rerank dominates retrieval, anti-kT v4 contributes a small
> but mechanistically interpretable selection-stage win on multi-hop bridge
> queries, and lock-in coherent paraphrase fails because LLM paraphrase
> guesses transfer ranking authority away from the original query.**

## Table 1 — HotpotQA-1k bridge (multi-hop), cs=384, terse

| selector | retriever | recall@5 | F1 | cit_acc | faith |
|---|---|---|---|---|---|
| greedy | dense | 0.8355 | 0.3967 | 0.7987 | 0.6225 |
| v4 anti-kT (α=0.7) | dense | 0.8355 | 0.3991 | **0.8126** | 0.6137 |
| MMR | dense | 0.8355 | 0.3940 | 0.7992 | 0.6224 |
| greedy + rerank | dense + bge-rrkv2-m3 | **0.9070** | **0.4157** | **0.8165** | 0.6289 |
| v4 + rerank | dense + bge-rrkv2-m3 | 0.9070 | 0.4146 | 0.8113 | 0.6304 |
| CLEAN-RAG | dense | 0.8355 | 0.2438 | 0.7181 | 0.4909 |

Δ-summary (vs greedy):
- v4 anti-kT: F1 +0.24pp, **cit_acc +1.39pp** (paired bootstrap n=1000, P=0.975)
- rerank: **recall@5 +7.15pp**, F1 +1.90pp, cit_acc +1.78pp
- v4 + rerank: NOT additive — F1 -0.11pp, cit_acc -0.52pp vs greedy+rerank.
  Rerank subsumes v4 (cross-encoder does the bridging job that v4's
  cluster topology approximates).
- CLEAN-RAG: catastrophic regression (-38% F1). Negative-with-explanation
  finding: radio CLEAN's "residual = new sources" assumption does not
  hold for RAG (relevant atoms are redundant, not orthogonal).

## Table 2 — NQ-open (single-hop), cs=384, terse

| selector | recall@5 | F1 | cit_acc | faith |
|---|---|---|---|---|
| greedy | 0.9790 | **0.7167** | 0.9592 | 0.5984 |
| v4 anti-kT | 0.9790 | 0.7145 | **0.9613** | 0.5996 |
| MMR | 0.9790 | 0.7144 | 0.9602 | 0.5978 |

All three are tied within noise. **Selector mechanism does not matter
when retrieval saturates and queries are single-hop.**

## Atomic interpretations

### Why v4 helps multi-hop only
The v4 anti-kT selector explicitly seeks two complementary leading jets and
uses a score-gated partner pull-in to ensure the second jet has enough
relevance to be an actual partner rather than noise. On bridge queries
(HotpotQA), the answer requires connecting two evidence atoms — exactly
the structure v4 enforces. On single-hop (NQ), one passage suffices, so
optimizing two-jet structure is moot.

The +1.39pp cit_acc gain on HotpotQA versus +0.21pp on NQ is therefore
exactly what the physics analogy predicts. The paper should frame v4 as a
*specialized* selector, not a universal one.

### Why rerank dominates retrieval
BGE-reranker-v2-m3 is a cross-encoder that scores (query, chunk) pairs
jointly, while BGE-M3 dense is a bi-encoder that scores them via cached
embeddings. The cross-encoder breaks the embedding-space bottleneck; for
HotpotQA's lexically-similar distractors, this matters by +7.15pp recall@5.

### Why terse prompt beat verbose
Verbose Qwen answers averaged ~50–80 tokens with full sentence structure;
gold answers are 1–5 token entities. F1 was capped near 0.12 by the
length mismatch. Terse prompt recovered 3.3× F1 (0.12 → 0.40). This is
not a selector improvement — it's removing a measurement ceiling.

### Why lock-in failed
Coherent sum of paraphrase queries pushes ranking toward LLM's *answer
prior* rather than the original query's information need. When the LLM
paraphrase guesses wrongly, retrieval moves toward the wrong direction by
construction (the √M boost amplifies whatever direction the paraphrases
point in, not necessarily the right one). Negative finding with
mechanistic explanation.

## Open questions / next experiments

1. **2WikiMultiHopQA / MuSiQue**: if v4 is multi-hop-specialized, the gain
   should *grow* on harder multi-hop tasks. Loader needed.
2. **v4 α-ablation** {0.3, 0.5, 0.7, 0.9}: characterize the partner-gate
   sensitivity curve — at what α does the bridging signal disappear?
3. **Selector × retriever cost-equivalence frontier**: with rerank cost
   measured, plot v4-on-dense vs greedy-on-rerank in (cit_acc, FLOPs) space.
   Defines when v4 is the right choice.
4. **CLEAN-RAG with sub-query decomposition**: replace the geometric
   residual with per-facet coverage residuals. (Substantial work, deferred.)

## Publication strategy

Path B (framework paper) — **a physics-inspired modular framework where
each selector mechanism has a documented failure mode, a documented
specialization, and a cost-equivalence frontier.**

Six characterized mechanisms:

| mechanism | analog | finding |
|---|---|---|
| greedy | baseline | reference |
| MMR | diversity | tied with greedy on bridge & single-hop |
| anti-kT v4 | jet clustering | +1.4pp cit_acc on multi-hop bridge, vanishes single-hop, subsumed by rerank |
| cross-encoder rerank | (no analog) | dominant mechanism; +7.2pp recall@5 |
| lock-in paraphrase | √M coherent sum | NEGATIVE — LLM paraphrase prior overrides query |
| CLEAN-RAG | Högbom CLEAN | NEGATIVE — radio "residual = new sources" assumption fails for RAG |

Each negative result has a sharp mechanistic explanation. Each positive
result has a *quantified specialization* (where it works, where it doesn't).
That's the rigor a Nature MI reviewer can credit.
