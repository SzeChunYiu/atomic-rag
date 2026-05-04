# Why v3 wins on synthetic IRC and fails on HotpotQA — full mechanistic account

**Date.** 2026-05-03
**Purpose.** Atomic-level mechanism, not headline numbers.

## Restatement of the algorithm in one sentence

v3 = greedy-by-score + **pull every jet partner of each pick**, where jets are anti-kT clusters of the candidate chunks.

In set notation, if `top_K(s)` are the highest-score chunks and `J(c)` is the
jet of chunk `c`, the v3 selection is:

$$
S_{v3} = \bigcup_{c \in \mathrm{greedy}} \bigl(\{c\} \cup J(c) \bigr)
$$

subject to token budget. The mechanism is a single design choice:
**append a chunk's clique to its inclusion**.

## The unifying principle that decides the regime

v3 helps if and only if:

$$
P(\text{chunk is gold} \mid \text{chunk is jet-partner of a gold pick}) > P(\text{chunk is gold} \mid \text{chunk is in candidate pool, not yet picked})
$$

i.e. clique membership must be a **stronger predictor of gold** than the candidate-pool prior.

When it's stronger: pulling cliques in adds gold. v3 wins.
When it's weaker: pulling cliques in adds noise, displacing budget that would have gone to higher-score (more-likely-gold) chunks. v3 loses.

This is a clean, falsifiable principle. Both regimes we observe agree with it.

## Synthetic IRC — why v3 wins, atomically

**Dataset construction.** 120 topics. Each topic produces multiple
near-paraphrase atoms intentionally engineered to share embeddings.
Distractors are 240 random unrelated texts. Gold pairs for each query
are *defined* to be atoms from the same topic.

**Clique structure.** Anti-kT pairwise distance is `min(s_i, s_j) * (1 −
cos(e_i, e_j))`. Two same-topic atoms have `cos ≈ 0.9+` → small `d_ij`
→ they merge into one jet. Distractors have `cos ≈ 0.1` to gold → large
`d_ij` → they end up in singleton jets.

So on synthetic IRC: **`J(c)` for a gold pick contains, almost
exclusively, more gold atoms.**

**Numerical evidence.** From iter 2 / v3 compare:
- Greedy mean gold-pair-coverage on synthetic = ~0.10
- v3 mean gold-pair-coverage on synthetic = ~0.31
- Δ = +0.21, n=100 paired (seed × cs), P=1.000, CI95 [+0.175, +0.242]

**Mechanistic translation.** Greedy alone often picks one half of a gold
pair (the one with higher SNR) and stops because the second half's SNR
is below the budget cutoff. v3's pull-in step grabs the second half via
clique membership. Hence the +0.21 boost is not magic — it's the
algorithm exploiting the dataset's by-construction clique structure.

## HotpotQA — why v3 fails, atomically

**Dataset structure.** Each query has gold = exactly 2 docs. The two
gold docs are typically **about different things** — connected only by
a bridge entity. Example:

> Q: "Alexander Kerensky was defeated and destroyed by the Bolsheviks
> in the course of a civil war that ended when?"
> Gold 1: "Alexander Kerensky" article (bridge entity)
> Gold 2: "Russian Civil War" article (answer entity, contains date)

The two gold docs **do not co-cluster** in embedding space. They share
the bridge entity but are otherwise about different topics.

**Clique structure.** Anti-kT clusters chunks by **topical** semantic
similarity. For a query about Kerensky and the civil war:
- Top-scoring chunks are mostly from the Kerensky article (gold 1) and
  one or two chunks from the civil-war article (gold 2).
- Anti-kT groups Kerensky-article chunks with each other AND with
  *non-gold* chunks that mention Kerensky (Wikipedia entities have
  many topical neighbors in the candidate pool).
- The civil-war chunks (gold 2) are in a *different* jet, alongside
  non-gold chunks about Russian history.

So on HotpotQA: **`J(c)` for a gold-1 pick contains mostly non-gold
topical neighbors**, and the second gold doc lives in a *different*
jet entirely.

**Atomic numerical evidence (cs=384, n=1000 queries):**

| | greedy-only | v3-only |
|---|---|---|
| chunks | 8483 | 8348 |
| of which gold | 316 (3.7%) | 128 (1.5%) |

Out of every 100 chunks v3 added beyond greedy, only 1.5 are gold.
Greedy's "extras" (chunks v3 didn't pick because v3 spent the budget on
partners) were 2.5× more likely to be gold.

**Why partners are *worse than random*.** This is the subtle bit. The
candidate pool has ~5–10% gold rate. v3-pulled partners have only 1.5%
gold rate. Why are partners *worse* than random candidates? Because:

1. The retriever already knows what's query-relevant. Top-K by retriever
   score is enriched in gold (~30% gold rate in top-5, ~10% in top-50).
2. Greedy already takes the top-K → those high-score chunks are no
   longer "available" partners.
3. **Available partners** = chunks in a clique with a top-K pick but
   *not themselves top-K*. By definition those scored *lower* than
   the threshold the retriever set for "likely query-relevant".
4. Topical similarity to a top-1 chunk is *correlated* with retriever
   score (top-1 chunk's clique members tend to mention similar entities
   → similar retriever scores). So the partners that aren't in
   greedy's top-K are, conditional on being in the cluster, the *low*
   retriever scorers — i.e. the *least* likely to be gold.

In one line: **v3-pulled partners are systematically the lowest-score
chunks in the most-relevant cluster.** That's exactly the wrong
selection signal.

**Downstream consequence.**

| | both gold docs covered | at least one gold |
|---|---|---|
| greedy cs=384 | 85.8% | 99.6% |
| v3 cs=384 | 68.7% | 98.0% |

The `at least one` rate stays high because the bridge-entity gold doc's
top chunk is almost always top-1 score and survives any reordering.
The `both` rate drops 17pp because v3 spends budget on bridge-doc
clique partners instead of on the answer-entity gold doc's chunks
(which live in a different cluster, get displaced).

This is why `citation_accuracy` drops by ~7% relative — v3's
selections cite chunks that are not the gold-2 doc's chunks, so they
fail the `cited_chunk_id ∈ gold_doc_chunks` test.

## Why MMR also failed to differentiate from greedy on HotpotQA

MMR with `λ=0.7` weights score 70%, diversity 30%. The score term
dominates because retriever scores have wide dynamic range while
diversity (max-cos-sim) is bounded in [0, 1]. So MMR's selection is
almost identical to greedy's, modulo a small reordering at the tail.

This is a known regime where MMR is essentially a no-op. To make MMR
bite we'd need `λ ≤ 0.3` or a stronger diversity term (e.g.
determinantal point process), but that's not v3 vs greedy — it's a
separate experiment.

## Why cs=64 / cs=128 had all selectors tied

Token budget = 1024. At cs=64, each chunk averages 64 tokens, so the
budget swallows ~16 chunks. At chunk_size ≤ 128 with 50 candidates,
the budget is large enough that all three selectors saturate it with
**the same set** of chunks — the budget never forces a real choice.
Selector mechanism cannot bite when there is no competition.

This is why we see selector signal only at cs ≥ 256 — the regime where
budget actually constrains selection.

## What v4 specifically fixes

v4 adds a gate: pull a partner only if `score(partner) >= max(α *
score(primary), median_floor)`. The mechanism this attacks:

- On synthetic IRC: partner scores are uniformly high within a gold
  cluster (paraphrases all match the query well). Most partners pass
  any gate `α ≤ 0.9`. Predicted: v4 ≈ v3 on synthetic.
- On HotpotQA: partners are by construction the *low-score* members of
  a cluster (per the analysis above). Most partners fail the gate.
  Predicted: v4 → greedy on HotpotQA, closing the −0.008 gap.

This is exactly the regime-aware behavior we want.

**The single most important question:** does any α exist where v4
preserves synthetic gain *and* matches greedy on HotpotQA? If yes,
publication-grade Pareto improvement. If no, the mechanism is
fundamentally regime-specific and we should publish the negative-as-positive
finding (the principle above) rather than a fix.

## Falsifiability

Each claim above is checkable:

1. **Clique-vs-gold correlation principle**: compute, for each query,
   `corr(c ∈ J(top_1), c ∈ gold)`. If this is positive on synthetic
   and ≤ 0 on HotpotQA, principle confirmed.
2. **Partner score < random**: compute, on HotpotQA candidates, mean
   score of `{partners of top-1 not in top-5}` vs mean score of
   `{random non-top-5 candidates}`. If partners < random, sub-principle confirmed.
3. **v4 regime-awareness**: pre-registered C2a, C2b, C2c, C2d (in
   atomic_diagnosis change log).

Items 1 and 2 are quick post-hoc verifications I should run on the
existing run dirs before drawing conclusions from v4.

## Empirical verification (HotpotQA cs=384, n=200 queries, 9000 partner draws)

| population | P(chunk is gold) | ratio vs random pool |
|---|---|---|
| top-5 by retriever SNR | 39.00% | 7.4× ↑ |
| random chunk in 50-cand pool | 5.30% | 1.0 (baseline) |
| **partner of top-1, not already in top-5** | **1.73%** | **3.1× ↓** |

The result is starker than expected. v3's partner pull-in is not just
*neutral* — it is **anti-correlated** with gold relative to the
candidate-pool prior. The retriever's score is a near-sufficient
statistic for query-relevance; chunks that scored low enough to fall
out of top-5 carry the implicit retriever judgment "not query-relevant".
v3 then pulls those chunks back in via topical similarity, **negating
the retriever's signal**.

This is the deepest level of the failure mode. The publication-grade
statement of the principle is:

> Anti-kT atomic-unit selection without score gating *recovers
> retriever-rejected chunks proportional to their topical similarity to
> retriever-accepted chunks*. On datasets where this recovery is
> beneficial (clique structure aligned with gold structure, e.g.
> synthetic IRC), the mechanism boosts gold-pair coverage by +0.21. On
> datasets where it is harmful (clique structure orthogonal to gold,
> e.g. multi-hop bridge queries), it pulls in topically-adjacent
> non-gold chunks at a rate 3× below the candidate-pool baseline,
> displacing genuine gold from the budget and reducing citation
> accuracy by ~7% relative.

The score gate in v4 is therefore precisely targeted: it filters out
exactly the *retriever-rejected partners* that cause the harm, while
preserving the *retriever-accepted partners* that drive the synthetic
gain.
