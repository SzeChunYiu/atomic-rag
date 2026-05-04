# P3 cs-sweep atomic diagnosis — v3 fails on HotpotQA via partner-dilution

**Date.** 2026-05-03
**Job.** LUNARC 2999682 (`acsrag-cs-sweep`, completed 34:43, exit 0)
**Configs.** cs ∈ {64, 128, 256, 384} × selectors ∈ {greedy, anti_kT v3 (n_jets=−2), mmr}.

## Summary table

| chunk size | citation_accuracy: greedy / v3 / mmr | answer_f1: greedy / v3 / mmr |
|---|---|---|
| cs=64  | 0.0869 / 0.0869 / 0.0869 (all tied, budget swallows) | 0.00675 / 0.00675 / 0.00675 |
| cs=128 | 0.0815 / 0.0814 / 0.0815 (essentially tied)         | 0.00617 / 0.00617 / 0.00617 |
| cs=256 | 0.0984 / **0.0908** / 0.0985 (v3 −0.008)            | 0.00948 / **0.00852** / 0.00948 (v3 −0.001) |
| cs=384 | 0.1143 / **0.1066** / 0.1144 (v3 −0.008)            | 0.01360 / **0.01303** / 0.01360 (v3 −0.0006) |

## Atomic-level smoking gun

For each query, compute `greedy_only = greedy_chunks − v3_chunks` and
`v3_only = v3_chunks − greedy_chunks`. Label each chunk gold-or-not via
`chunk_id → doc_id ∈ gold_doc_ids`.

| metric | cs=256 | cs=384 |
|---|---|---|
| queries with diverging selections | 1000/1000 | 1000/1000 |
| greedy-only chunks (total / gold) | 8956 / 376 (**4.2%**) | 8483 / 316 (**3.7%**) |
| v3-only chunks (total / gold) | 8772 / 123 (**1.4%**) | 8348 / 128 (**1.5%**) |
| both gold docs covered (greedy / v3) | 87.6% / **70.7%** | 85.8% / **68.7%** |
| at-least-one gold doc (greedy / v3) | 99.9% / 98.7% | 99.6% / 98.0% |

**v3's "extras" are 2.5× less likely to be gold than greedy's.** v3 loses
17 percentage points of full-gold-coverage on both cs=256 and cs=384.

## Atomic root cause (which atom failed)

v3 = atomic-unit greedy + **pull all jet partners** of each primary pick.

- **Synthetic IRC** (where v3 wins +0.21): partners are gold by
  construction (dataset has clusters of gold atoms with high pairwise
  similarity). Pull-in sweeps in extra gold → mechanism amplifies signal.
- **HotpotQA** (where v3 loses −0.008): gold = exactly 2 docs. Anti-kT
  jets cluster *topically*, not by ground-truth label. Most jet partners
  are topical-similar but **non-gold**. Pull-in displaces real gold
  chunks (specifically the second-gold-doc chunks, the answer-entity
  doc, since the bridge doc is top-1 score and survives reordering) with
  topical noise.

The mechanism is exactly inverted between the two regimes.

## "At-least-one" stays at 98–99% — why

The bridge entity in a HotpotQA bridge query is almost always the
top-score chunk. Both selectors keep it. The dilution lives at chunk 5+
of the budget where the second gold doc lives — that's where v3 swaps
real gold for topical partners.

## Implication for the publication

This is **publishable as a negative-on-HotpotQA + positive-on-synthetic
result** — both regimes characterize the mechanism completely. The
paper's claim becomes:

> Anti-kT atomic-unit selection beats greedy when partner-coherence
> tracks ground-truth structure (synthetic IRC: +0.21, n=100, P=1.000),
> and is dominated by greedy when partners are merely topically similar
> (HotpotQA: −0.008 cit_acc, n=1000). The crossover is governed by the
> partner-score distribution.

This is honest and informative. It also motivates v4 directly.

## v4 design — score-gated partner pull-in

**The atomic change.** A jet partner is pulled in only if its retriever
score passes a gate:

```
score(partner) ≥ max(score(primary) * α, median(candidate_scores))
```

- α ∈ [0, 1]: relative gate against the primary's score.
  - α = 1: only equal-or-better partners (very strict, ≈ greedy).
  - α = 0: all partners (= v3).
- median floor: absolute floor — never pull a clearly-bad partner.

**Why this should preserve synthetic + close HotpotQA gap:**
- Synthetic IRC: partners co-cluster with primary → high score → pass
  gate → preserve +0.21.
- HotpotQA: low-score topical partners fail gate → behavior reduces
  toward greedy → close −0.008 gap.

**Pre-registered claims (paired bootstrap, n=100/1000).**
- C2a: v4 ≥ v3 on synthetic IRC at α ∈ {0.5, 0.7, 0.9} — IRC win preserved.
- C2b: v4 ≥ greedy on HotpotQA cs=256 cit_acc at α ∈ {0.5, 0.7, 0.9}.
- C2c: v4 ≥ greedy on HotpotQA cs=384 cit_acc at α ∈ {0.5, 0.7, 0.9}.
- C2d: There exists α* such that v4(α*) Pareto-dominates greedy across
  both synthetic and HotpotQA.

If C2d passes → publication-grade Pareto improvement. If C2a passes
but C2b/C2c fail → the gate isn't enough, deeper rethink needed. If
C2a fails → atomic-unit framing has fundamental issue, fall back to v2.
