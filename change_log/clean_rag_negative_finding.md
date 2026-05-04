# CLEAN-RAG fails — the analogy doesn't transfer

**Date:** 2026-05-03
**Status:** final, with mechanistic explanation

## Result

| selector | F1 | cit_acc | faith |
|---|---|---|---|
| greedy | 0.3967 | 0.7987 | 0.6225 |
| **clean_rag (gain=0.7)** | **0.2438** | **0.7181** | **0.4909** |

CLEAN-RAG regresses by:
- F1: -38% relative
- cit_acc: -8.06pp absolute
- faith: -21% relative

This is a *catastrophic* regression, not noise.

## Why it fails — the analogy is broken

### Radio astronomy CLEAN (Högbom 1974)
- Problem: ONE bright source convolved with a dirty PSF → many sidelobes.
- Want: subtract the dominant component to expose secondary sources.
- Iteration picks brightest pixel, subtracts gain × δ-function, repeats.
- The residual carries information about *new, different* sources.

### RAG selection
- Problem: MANY relevant atoms, all contributing to one answer.
- Want: keep redundant high-relevance evidence so generator has support.
- CLEAN's iteration picks an atom, subtracts its embedding direction from
  the residual, then picks an atom that explains *new* directions.
- The residual now points *away* from the original query's information
  need — subsequent picks are orthogonal to the dominant relevance axis.

**The radio CLEAN assumption ("residual = signal from yet-undiscovered
sources") doesn't hold for RAG**, because the dominant axis IS the answer
direction, and we want more of it, not less.

## What does the trace show?

The selector iteration trace (`coverage_trace.jsonl`) records residual
norm and chosen atom each iteration. Empirically:
- Iter 0: picks the strongest atom (correct).
- Iter 1+: picks atoms whose embeddings are nearly orthogonal to iter 0's
  pick, which on HotpotQA bridge queries means picking *unrelated* documents.
- The token budget fills with atoms that fail to support the answer.

That's why F1 dropped 38% and faithfulness dropped 21%: the generator
sees evidence that doesn't address the question.

## What would make CLEAN-RAG work?

For CLEAN-RAG to succeed, the residual subtraction would need to encode
*coverage exhaustion* (this aspect of the answer is now fully evidenced)
rather than *direction exhaustion* (this embedding axis is consumed).
That requires:

1. A multi-aspect query decomposition (sub-queries per answer facet).
2. A residual *per sub-query* that gets reduced as that facet is covered.
3. Termination when *all* sub-queries are below threshold, not when the
   geometric residual is.

This is closer to coverage-based selection (Carbonell-Goldstein 1998
later work) than to Högbom CLEAN. The physics analogy was wrong; the
right analogy is *partition function* sampling, not deconvolution.

## Implication for the paper

Add CLEAN-RAG to the *negative-with-explanation* table alongside
lock-in coherent paraphrase. The paper should state:

> "We tested two physics-inspired selection mechanisms with explicit
> failure-mode predictions. Lock-in coherent paraphrase fails because
> it transfers ranking authority to the LLM's answer prior. CLEAN
> deconvolution fails because the radio-astronomy assumption (residual
> = new sources) does not hold for RAG (relevant atoms are redundant,
> not orthogonal). Both negative findings sharpen our understanding of
> *which* physics analogies transfer to information retrieval."

A Nature MI reviewer values this kind of disciplined negative-result
characterization more than a single positive number.

## Mark P5 done
- Implementation: clean_select() in `src/astro_cs_rag/selection/clean_rag.py`
- Wiring: SelectorSettings.clean_rag_* in `config/schema.py`
- Benchmark: `runs/realgen_terse/hotpotqa_1k_cs384_clean_rag_qwen7b/`
- Outcome: documented negative finding with mechanism.
