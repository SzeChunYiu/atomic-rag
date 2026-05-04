# P3 — anti-$k_T$ v3 (atomic-unit greedy)

## Why a v3 was needed

The v1 → v2 history (see `P3_negative_finding_jet_select_v1.md`):

| selector | synthetic IRC (mean cov) | HotpotQA cs=512 (citation acc) |
|---|---|---|
| greedy | 0.539 | 0.886 |
| anti-$k_T$ v1 (n_jets=1, merge order) | 0.725 (+0.186) | 0.014 (collapse) |
| anti-$k_T$ v2 (n_jets=-1, SNR sort) | 0.539 ($\equiv$ greedy) | 0.886 ($\equiv$ greedy) |

v1 *won* on the synthetic stress test but *catastrophically failed* on real
HotpotQA. v2 *fixed* HotpotQA but flattened the synthetic gain — a complete
no-op vs greedy.

Diagnosis: v1's win came from a specific mechanism — packing leading-jet
members in **anti-$k_T$ merge order**, which encodes joint-evidence
affinity. SNR-sorting jet members (v2) breaks the gold-pair atomicity that
makes the leading jet a useful selection unit on joint-evidence problems.

## v3 specification (`n_jets = -2`)

`selection/jet_select.py::_select_v3_atomic_unit`:

1. Cluster candidates with anti-$k_T$ (unchanged: $d_{ij} = \min(s_i^{-2},
   s_j^{-2}) \cdot \Delta_{ij}^2 / R^2$).
2. Build `chunk -> jet_index` and `jet_index -> ordered_members`.
3. Greedy pack chunks by SNR descending. When chunk $c$ is selected as
   *primary*, immediately also include **all jet partners** of $c$ — i.e.,
   every other chunk in the same jet — in anti-$k_T$ merge order.
4. Continue greedy with remaining unseen chunks until the budget fills.

Token budget enforcement: jet partners are dropped if they would overflow
the budget; a primary is preferred over partners (greedy ordering is the
soft preference).

## Synthetic comparison (job 2998831)

3 seeds × 12 chunk_sizes ∈ [50, 300] × 4 variants on 120-topic /
240-distractor synthetic IRC corpus (paired bootstrap, 10k resamples):

| comparison | mean diff | P(a > b) | CI95 |
|---|---|---|---|
| **v3 vs greedy** | **+0.2155** | **1.000** | [+0.161, +0.266] |
| v1 vs greedy | +0.1956 | 1.000 | [+0.143, +0.245] |
| v3 vs v1 | +0.0199 | 1.000 | [+0.016, +0.024] |
| v2 vs greedy | 0.0 | 0.0 | [0, 0] |

**v3 strictly Pareto-dominates greedy AND v1 on synthetic.** The +0.020
gap over v1 is small but tight (CI doesn't cross 0); intuition is that v3
also benefits from non-leading-jet partner pull-ins when the greedy primary
falls in jet 2/3 (v1 ignores those).

## Why v3 should not collapse on HotpotQA

At cs=512 with BGE-M3 dense retrieval on HotpotQA, candidates are large
distinct chunks. Cosine similarities between distinct chunks are typically
$\le 0.5$, well below the $R = 1.0$ merge threshold for medium-SNR atoms.
Most chunks therefore form **singleton jets** — no partners to pull in.
v3 then degenerates to greedy by construction, recovering the 0.886
citation accuracy.

The mechanism advantage manifests at *small* chunk sizes where same-doc
chunks of joint evidence cluster tightly — exactly the regime targeted by
the chunk-size sweep (job 2998872).

## Pre-registered prediction for the cs-sweep

Job 2998872 (`run_chunksize_sweep.slurm` with v3 configs) tests
HotpotQA-1k × {cs64, cs128, cs256, cs384, cs512, cs768} × {greedy, v3, mmr}.

Pass conditions:

- **C1c (no regression)**: v3 citation_accuracy $\ge$ greedy at every
  chunk size. Required.
- **C1d (real-data IRC win)**: v3 citation_accuracy $\ge$ greedy + 0.01
  at cs $\le$ 128. Required to claim real-data transfer of the IRC
  mechanism.

If C1c passes but C1d fails, the paper repositions: anti-$k_T$ v3 is a
*safe replacement* for greedy with provable IRC properties on synthetic;
it does not (yet) demonstrate practical advantage on HotpotQA-style multi-hop
QA at production chunk sizes.

If both pass, the paper claims chunk-size-bounded real-data transfer of
the synthetic IRC mechanism.

## Files changed

- `src/astro_cs_rag/selection/jet_select.py` — add `_select_v3_atomic_unit`
  helper; dispatch on `n_jets == -2`; revert SNR-sort for `n_jets >= 1`
  (preserves v1's merge-order mechanism).
- `configs/benchmarks/chunksize_sweep/*.yaml` — regenerated with
  `anti_kt_n_jets: -2`.
