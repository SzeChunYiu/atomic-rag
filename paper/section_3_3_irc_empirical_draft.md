# Section 3.3 — Empirical IRC robustness of anti-$k_T$ evidence-jet selection

*Draft v1 (synthetic stress-test results only; HotpotQA chunk-size sweep pending job 2998104).*

## 3.3.1 Synthetic stress test design

To isolate the chunk-boundary failure mode without the confound of retrieval
noise, we constructed a synthetic corpus where every gold answer is a
*joint* claim spanning two sentences:

- 120 gold documents. Each document is exactly two sentences:
  *"$T$ is associated with $A$. In particular, observations of $T$ reveal $B$."*
  Selecting only sentence 1 yields the topic $T$ and predicate $A$ but misses
  the answer phrase $B$; selecting only sentence 2 yields $B$ but no anchor.
  Only the *joint* selection contains the full answer.
- 240 distractor passages on disjoint topics (linear algebra, polymer chains,
  queueing theory, etc.) — chosen so retrieval scores cluster on topic words
  rather than gold sentences leaking into distractors.
- 120 queries of the form *"What does $T$ reveal about its physical
  mechanism?"*, with the gold answer phrase $B$ stored in metadata.

The benchmark sweeps:

- **20 chunk sizes** (in characters), uniformly spaced in $[50, 300]$. At
  $\le 76$, the gold pair always splits across $\ge 3$ chunks. At $\ge 168$
  both sentences typically land in a single chunk.
- **4 selectors:** `greedy` (top-$k$ by SNR), `mmr` (Carbonell–Goldstein 1998
  with $\lambda = 0.7$), and **anti-$k_T$ v3** (this paper, $R = 1.0$ in cosine
  units, atomic-unit greedy: greedy by SNR with all jet partners pulled in
  for each primary selection in anti-$k_T$ merge order). v1 (leading-jet
  only) and v2 (multi-jet, SNR-sorted) variants are reported in §3.3.7
  as ablations.
- **5 random seeds** (regenerated distractor pool, deterministic gold).

Total: $300$ benchmark runs, $\sim 12{,}000$ query evaluations.

The selection-sensitive metric is **gold-pair coverage**: the fraction of
queries whose selected context contains both sentences of the gold document
(detected by substring match of the metadata-stored answer phrase $B$).

## 3.3.2 Headline result

| selector | mean coverage | stdev | CV |
|---|---|---|---|
| **anti-$k_T$ v3** | **~0.747** | **~0.130** | **~0.174** |
| anti-$k_T$ v1 | 0.725 | 0.135 | 0.186 |
| greedy | 0.539 | 0.169 | 0.314 |
| anti-$k_T$ v2 | 0.539 | 0.169 | 0.314 (degenerate) |
| mmr | 0.006 | 0.007 | (failure) |

Anti-$k_T$ v3 strictly Pareto-dominates greedy and v1: $+0.208$ absolute
($\sim$39% relative) over greedy, and $+0.022$ over v1, on gold-pair
coverage. MMR collapses to near-zero everywhere — it explicitly
de-prioritizes near-duplicate chunks, but the joint gold pair is two
halves of the *same* document and looks redundant by cosine similarity.
We treat MMR as the anti-baseline showing that *any* repulsive selector
is wrong-by-design for joint-evidence problems.

## 3.3.3 Statistical significance

We test the headline claim with a paired-bootstrap on per-(seed, chunk_size)
gold-pair-coverage differences (10 000 resamples, n=100 paired observations
across 5 seeds × 20 chunk sizes):

| comparison | mean diff | P(a > b) | CI95 |
|---|---|---|---|
| **anti-$k_T$ v3 vs greedy** | **+0.2084** | **1.000** | [+0.175, +0.242] |
| anti-$k_T$ v1 vs greedy | +0.1862 | 1.000 | [+0.153, +0.219] |
| anti-$k_T$ v3 vs v1 | +0.0223 | 1.000 | [+0.019, +0.025] |
| anti-$k_T$ v2 vs greedy | 0.0 | 0.0 | [0, 0] |

**v3 strictly Pareto-dominates greedy with overwhelming evidence:** the
95% CI on the per-(seed,cs) mean difference is entirely above zero by
more than ten standard errors. v3 also strictly beats v1 by a small but
unambiguously positive margin (CI excludes zero by 19$\sigma$). v2 is a
degenerate no-op vs greedy; we discuss why in §3.3.7.

**C1b — anti-$k_T$ has lower chunk-size variance than greedy.** Bootstrap
on within-seed standard deviation across the 20 chunk sizes (5 000
resamples):

- $P(\text{stdev}_{\text{anti-}k_T} < \text{stdev}_{\text{greedy}}) = 0.787$
- mean stdev difference: $-0.032$
- 90% CI: $[-0.100, +0.039]$ — crosses zero

The IRC stability claim is *directional* (anti-$k_T$ is more chunk-stable on
average) but not significant at the conventional $p < 0.05$ threshold for
the magnitude of the dataset. The mean-coverage claim (C1a) carries the
weight; the variance claim (C1b) is a secondary observation that the
clustering smooths but does not eliminate chunk-size sensitivity.

## 3.3.4 Mechanism

The per-chunk-size profile reveals a U-shape:

- **cs $\le 76$ — the boundary regime.** The gold pair always splits across
  $\ge 3$ chunks. Greedy picks top-1 by SNR; the second half typically has
  slightly lower SNR (less topic-word overlap) and falls outside the budget.
  Anti-$k_T$'s clustering aggregates the same-document chunks into one jet:
  selecting the leading jet fetches both halves. Greedy's coverage bottoms
  out at $0.36$; anti-$k_T$'s at $0.43$.
- **cs $\in [89, 116]$ — the sweet spot.** A single chunk *almost* contains
  the joint pair; one selector pick gets you both. Anti-$k_T$ peaks at
  $0.83$–$0.88$; greedy at $0.62$–$0.66$.
- **cs $\ge 168$ — the convergence regime.** Both halves fit comfortably
  in a single chunk; both selectors plateau near $0.78$. Anti-$k_T$ does
  not regress.

The U-shape *is* the IRC-safety signature. The algorithm matches greedy at
chunk-size extremes where boundary effects cannot bite, and outperforms
where they do. The variance-stability claim (C1b) is a side effect: a
selector whose boundary regime is closer to its sweet spot has a smaller
stdev across chunk sizes.

## 3.3.5 What this does not prove

This is a *synthetic* stress test, deliberately constructed to expose the
chunk-boundary failure mode. The benchmark cannot tell us:

1. Whether the same effect transfers to real-world QA, where gold spans are
   not constructed to cluster tightly.
2. Whether the mechanism survives in pipelines with reranking,
   query rewriting, or generator post-processing.
3. The magnitude of the gain at production-scale chunk sizes
   (HotpotQA's typical $\text{cs} = 512$).

§3.3.6 reports the corresponding HotpotQA-1k chunk-size sweep, which we
pre-registered to address (1) and (3) directly.

## 3.3.6 Real-data: HotpotQA chunk-size sweep [PENDING — job 2998104]

*[To fill: HotpotQA-1k under BGE-M3 dense retrieval, chunk_size $\in \{64,
128, 256, 384, 512\}$, selectors {greedy, anti-$k_T$ v2, mmr}.
Expected: anti-$k_T$ v2 outperforms greedy at small cs (boundary effects)
and matches at large cs (no regression). If small-cs gain holds with
$p < 0.01$, C1c/C1d pass. If not, paper repositions the IRC claim as
synthetic-only with explicit caveat.]*

## 3.3.7 Selector design history (v1 → v2 → v3)

The selector underwent three design iterations against this benchmark and
real HotpotQA. We report all three to make the design space transparent.

**v1 ($n_\text{jets} = 1$, members in anti-$k_T$ merge order).** Only the
leading jet's members are eligible for the budget. Wins on synthetic
(+0.186 over greedy) because the gold pair clusters into the leading jet
by construction, and the merge order encodes joint-evidence affinity:
the two gold halves enter the budget before less-related members.
*Catastrophic on real HotpotQA at cs=512:* citation accuracy collapses
$0.886 \to 0.014$. Multi-hop gold spans heterogeneous documents that
cluster less tightly than near-duplicate distractors; the leading jet by
summed relevance is then a distractor cluster that excludes gold.

**v2 ($n_\text{jets} = -1$, all jets in relevance order with members
sorted by SNR descending).** Designed to fix v1's HotpotQA collapse.
Reduces to greedy by SNR whenever the leading jet's highest-SNR member
already dominates the budget. Restores citation accuracy to $0.886$ on
HotpotQA, *but degenerates to greedy on synthetic too*: the SNR-sort
within jets breaks the merge-order atomicity that made v1 work. **v2 is
a no-op.** We report it as a cautionary tale: a fix that "passes" by
making the method stop differing from the baseline does not fix anything.

**v3 ($n_\text{jets} = -2$, atomic-unit greedy).** Greedy by SNR is the
primary ranking; for each chunk selected as primary, *all* of its jet
partners are pulled in (in anti-$k_T$ merge order) before greedy advances
to the next primary. This preserves greedy's ranking signal AND exploits
the clustering as an atomic-unit selection mechanism. The IRC theorem is
unchanged — the clustering decision is still anti-$k_T$ — and the
selector is now a *soft preference* on top of greedy rather than a hard
exclusion. v3 wins on synthetic ($+0.208$ over greedy, $+0.022$ over v1)
and is expected to match greedy on HotpotQA at production cs (singleton
jets → no partner pull-in → greedy by construction).

We report v3 throughout. The v1 collapse and the diagnostic trail are
documented in `change_log/P3_negative_finding_jet_select_v1.md`. The v3
design and pre-registered claims are documented in
`change_log/P3_v3_atomic_unit_greedy.md`.
