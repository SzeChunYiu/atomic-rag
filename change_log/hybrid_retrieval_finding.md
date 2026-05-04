# Hybrid sparse+dense retrieval — bridge-recall improvement (2026-05-03)

## Setup

Atom-level retrieval, HotpotQA-1k, comparing five methods:
- dense: BGE-M3 cosine alone
- bm25: BM25 (RM3-style) alone
- linear_a0.5 / linear_a0.7: dense·α + BM25·(1−α) on z-normalized scores
- rrf: reciprocal-rank fusion (Cormack et al. 2009)

R@K = recall (any gold doc in top-K atoms);
Br@K = bridge recall (BOTH gold docs present in top-K).

## Results

| Method | R@10 | R@30 | R@50 | R@100 | Br@10 | Br@30 | Br@50 | Br@100 |
|---|---|---|---|---|---|---|---|---|
| dense (BGE-M3) | 0.844 | 0.911 | 0.933 | 0.962 | 0.697 | 0.824 | 0.867 | 0.924 |
| bm25 | 0.768 | 0.873 | 0.901 | 0.932 | 0.567 | 0.756 | 0.809 | 0.871 |
| linear_a0.5 | 0.812 | 0.904 | 0.926 | 0.949 | 0.644 | 0.813 | 0.858 | 0.899 |
| linear_a0.7 | 0.839 | 0.916 | 0.939 | 0.962 | 0.691 | 0.836 | 0.880 | 0.923 |
| **RRF** | **0.851** | **0.928** | **0.948** | **0.967** | **0.711** | **0.858** | **0.895** | **0.933** |

## Key findings

1. **RRF beats every other method at every K**, including dense alone.
2. **Bridge recall is the bigger gain than overall recall** — at K=50,
   Br improves +2.8pp (0.867 → 0.895), R improves +1.5pp (0.933 → 0.948).
   At K=30, Br improves +3.4pp.
3. BM25 alone underperforms dense (atoms are short, BM25 has less
   signal at sentence level), but provides complementary information
   captured by fusion.
4. Linear fusion is sensitive to α; RRF is rank-based and parameter-free.

## Mechanism

The B2 bottleneck is bridge-doc recall: queries miss the second gold
document because BGE-M3's dense similarity rewards topic match but
underweights sparse-but-discriminative entity terms. BM25 amplifies
exact-token matches on rare entities. Their fusion gets both:
topic-level dense neighborhood + entity-level lexical match. Bridge
queries are exactly the case where the second gold doc shares a rare
entity name with the query (the bridge entity), which BM25 catches.

## Implication

This is a B2 contribution — bridge recall improvement at zero GPU
cost. The full paper-pipeline impact depends on whether the ~3pp
candidate-pool gain translates to F1 after selection and generation,
which the queued GPU experiments will tell us. RRF candidate-pool
will be wired into the multiscale + maxent + replicas stack as a
drop-in replacement for dense-only retrieval.

Combined with the queued multi-scale RG (which also addresses B2 via
scale-dependent kernels), we have two complementary mechanisms hitting
the same bottleneck. If both improve F1 independently, that's strong
ablation evidence; if they don't compose, the more-effective one wins.

## Falsifier

If RRF candidate-pool gain disappears after selection (i.e., MaxEnt
or submodular selects from the wider pool but ends up with the same
final atoms), then the gain is at retrieval but not at generation —
the bottleneck is downstream. Will be measured directly in the next
GPU experiment.
