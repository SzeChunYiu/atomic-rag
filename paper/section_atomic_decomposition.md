# Atomic decomposition of retrieval-augmented generation

## Motivation

A retrieval-augmented generation (RAG) pipeline is a composite system.
End-to-end metrics (answer F1, citation accuracy) confound effects from
many stages: chunking, retrieval, reranking, selection, prompt
assembly, generation, citation. When one number drops, the standard
research move is to hill-climb the system as a whole — try a new
retriever, swap a generator, change a prompt — and report whichever
combination gives the best aggregate score.

This conflates *what improved* with *why it improved*. It also makes
unsuccessful interventions uninformative: an intervention that does
nothing on the headline metric may have helped one stage and hurt
another. The composite cancellation is invisible.

We propose an alternative: **decompose the pipeline into discrete
atomic stages, attribute the loss to specific stages with per-query
diagnostics, and pick interventions only at stages where the loss
actually lives**. This is the standard methodology in instrumented
physical experiments (where each detector subsystem is calibrated
separately) and in compiler optimization (where stage-by-stage
profiling identifies hot spots). It is not standard in RAG.

## The eleven atomic stages

We define the canonical RAG pipeline:

1. **Chunking** — document → fixed-size chunks
2. **Chunk embedding** — chunk → dense vector
3. **Query embedding** — query → dense vector
4. **Top-K retrieval** — cosine similarity, partition, sort
5. **Reranking** — cross-encoder rescoring of top-K
6. **Selection** — pick a token-budget-respecting subset
7. **Prompt assembly** — interleave query + selected evidence
8. **Generation** — LLM produces answer with inline `[E_i]` citations
9. **Chain-of-thought extraction** — strip reasoning from final answer
10. **Citation extraction** — parse `[E_i]` → atom IDs
11. **Metric evaluation** — F1, EM, citation accuracy

Each stage has a measurable input and output. Stage-level loss is
defined as the maximum metric improvement achievable by *only* fixing
that stage, with all others held fixed.

## Per-query joint failure analysis

For each query, we record four boolean indicators on a held-out set:

- F1+ : answer F1 ≥ 0.5
- cit+ : citation accuracy ≥ 0.5
- gold+ : at least one gold document represented in the selected pool
- text+ : the gold answer text appears within at least one selected atom

The 16-cell joint distribution of these indicators identifies *exactly*
which stages are responsible for each query's failure. Empty cells
indicate that some failure modes are not present in the data; large
cells indicate which interventions will move the headline metric.

For HotpotQA-1k with our atom-level pipeline (D04+D06 with CoT), the
distribution is:

| F1 | cit | gold | text | Count | Interpretation |
|---|---|---|---|---|---|
| + | + | + | + | 310 | All stages succeeded |
| − | + | + | + | 139 | **Generation bottleneck**: support is present, model fails to extract |
| − | − | + | + | 124 | Selection picked gold doc but model didn't use it |
| − | − | + | − | 119 | Span selection failure: gold doc selected but right sentences not |
| + | − | + | + | 107 | Right answer despite hallucinated citations |
| − | + | + | − | 82 | Cited correctly but no answer text in pool |
| + | + | + | − | 79 | Right answer + cite from prior model knowledge |
| + | − | + | − | 39 | Right answer with no support (closed-book) |
| − | − | − | − | 1 | Catastrophic full miss |

Three observations follow immediately:

**1. F1 and citation accuracy are nearly independent.** The Pearson
correlation across 857 queries with non-zero values is 0.054. Citation
hallucination and answer correctness are decoupled failure modes; an
intervention that improves one need not improve the other.

**2. Generation is the single largest bottleneck.** 26.3% of queries
have the gold answer text *in the selected pool* but F1 < 0.5 — the
model has the answer in its context and fails to extract it.

**3. Bridge recall (B2) accounts for 14.7% of queries.** This matches
the published difficulty of HotpotQA bridge questions and isolates a
retrieval-stage problem distinct from generation.

## Bottleneck table

| Bottleneck | Stage | Queries affected | Mechanism |
|---|---|---|---|
| B1 | Selection | ~10% | objective imbalance: coverage dominates relevance |
| B2 | Retrieval | 14.7% | bridge entity not visible at single-scale dense |
| B3 | Citation | n/a (cit_acc, not F1) | LLM hallucinates `[E_i]` IDs |
| B4 | Generation | 26.3% | position bias / distraction in long contexts |
| Saturated | various | ~31% | all stages succeed |
| Partial | various | ~17% | mixed/partial failures |

## Why this changes how we pick interventions

Without the decomposition, we would naturally try methods proportional
to stage size: many retrieval methods, many selection methods, fewer
generation methods (because retrieval and selection are easier to
modify than generation). The decomposition reverses this: B4
(generation) is the largest single bottleneck and is therefore the
best place to invest, even though it is the hardest stage to modify.

Conversely, B1 (selection) is mid-sized and well-suited to algorithmic
intervention. B2 has multiple complementary mechanisms (bridge entity
recall via PRF, RG-style multi-scale retrieval, hybrid BM25+dense
fusion). B3 turns out to be a pure structural failure with a trivial
fix (Section 4) — investing more there is wasteful.

## Falsifiability of each stage's claim

For an intervention at stage *k* to claim a real effect, it must:

1. Move the metric on the queries assigned to stage *k* by the
   diagnostic, not just the headline metric.
2. Not regress other stages.

We apply this test to every method in the rest of the paper. Methods
that improve the headline F1 without improving the assigned-stage
fraction are flagged as confounded; methods that improve the assigned
fraction but not the headline are documented as stage-isolated and
re-evaluated for downstream amplification.

## Comparison to existing methodologies

Our framework differs from existing RAG diagnostic work in three ways:

1. **Per-query joint indicators**, not aggregate stage metrics. Aggregate
   metrics conflate failure modes (e.g., recall@K does not distinguish
   missing-bridge from missing-single).
2. **Stage-conditional intervention claims**. Each method is required to
   move the diagnostic-assigned stage, not just the headline.
3. **Failure-bucket-driven method selection**, not method-driven failure
   analysis. We choose which method to try based on which bucket
   dominates, not which method we happened to read about recently.

The methodology applies to any RAG variant with the same eleven-stage
canonical structure. We expect it to generalize across datasets and
generators, though the bucket distribution will shift.
