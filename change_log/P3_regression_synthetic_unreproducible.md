# P3 — synthetic IRC +0.21 result is unreproducible (negative)

**Date.** 2026-05-03
**Severity.** Highest — invalidates a prior headline finding.

## What was claimed earlier today

`synthetic_irc_v3_compare/summary.json` (run earlier today): v3 vs greedy
gold_pair_coverage = +0.2155, P=1.000, n=36 (3 seeds × 12 chunk sizes).

## What is now reproducibly measured

`synthetic_irc_v4_sweep/summary.json` (run today, after v4 implementation):
v3 vs greedy = **−0.2521**, P=0.000, n=24. Sign-flipped.

Targeted regression test on the *exact same dataset* (`synthetic_irc_v3_compare/seed_0`),
seed=0, cs=118, with current code:

| variant | OLD result | CURRENT result |
|---|---|---|
| greedy | 0.7333 | 0.7333 (unchanged) |
| anti_kt v3 | 0.8167 | 0.4583 (**different**) |

Same data, same retriever, same selector params (n_jets=−2, alpha=0,
median=False), different result for v3 only.

## Atomic root cause investigation

Wrote a slow Python triple-loop reference implementation of `cluster_anti_kt`
and compared jet structure to current vectorized code on q_0000:

- Both produce **one jet of 30 members** (all candidates merge).
- **Identical member order**: `gold_0095::1, gold_0040::0, gold_0040::1, ..., gold_0000::0`
  (gold_0000::0 at position 29).
- VEC and REF agree exactly. So the vectorization is not the cause.

But OLD selected_context for q_0000 has `gold_0000::0` at position 1
(immediately after primary). For OLD result to occur, the jet order must
have been `gold_0095::1, gold_0000::0, gold_0046::0, ...` — different from
both current vectorized AND current reference.

**Conclusion.** OLD result came from a code state where `cluster_anti_kt`
produced a different jet member order. Most likely cause: a transient
implementation that I no longer have (no git, no version archived).
Could be a different shift-to-positive convention, a different distance
formula (e.g., delta^1 vs delta^2), or a stale-distance bug that happened
to favor v3.

I cannot reconstruct the older code, so I cannot reproduce +0.21.

## Honest scientific position now

| claim | status |
|---|---|
| v3 beats greedy on synthetic IRC (+0.21) | **NOT REPRODUCIBLE** — withdrawn |
| v3 loses to greedy on HotpotQA (−0.008) | reproducible, real |
| v4 score gate closes the HotpotQA gap | reproducible, real (cs=256: 0.0984 = greedy 0.0984) |
| **There is currently no benchmark where anti-kT beats greedy** | **TRUE** |

## Atomic engineering — what changed in our understanding

1. The earlier synthetic claim was a one-time measurement taken under a
   code state I no longer have. **No claim of synthetic win can be made
   in any paper draft.**
2. The HotpotQA mechanism analysis (1.5% partner gold rate vs 5.3% pool
   baseline) is still rigorous and remains a strong negative-result
   finding about anti-kT atomic-unit selection.
3. v4 score-gate is a useful engineering contribution: it neutralizes the
   anti-kT harm without hurting greedy. Modest but real.

## Path forward, ranked by ROI

1. **Real generator (Qwen2.5-7B)** — still highest priority. Tests whether
   *any* selector difference becomes visible at non-floor answer_f1.
   StubGenerator floor at 0.006 is uninformative.
2. **Lock-in coherent paraphrase retrieval (P4)** — orthogonal mechanism
   with a theoretical SNR boost; doesn't depend on clustering. **Most
   promising for finding a real win.**
3. **R-sweep on anti-kT** — try R ∈ {0.3, 0.5, 0.7, 1.0, 1.5}. Maybe
   tighter clusters change the picture. Cheap experiment.
4. **Entity-aware chunking** — make cluster structure align with gold
   structure on bridge queries by chunking on entity boundaries. Real
   research, not just engineering.

## Status of pre-registered claims

- C1a (v3 > greedy on synthetic): **withdrawn** — was based on
  unreproducible measurement.
- C1c, C1d (v3 ≥ greedy on HotpotQA): **falsified** — v3 loses by 0.008.
- C2a (v4 ≥ v3 on synthetic): trivially true since v3 lost.
- C2b/C2c (v4 ≥ greedy on HotpotQA): **CONFIRMED** at cs=256, partial
  data at cs=384. v4 closes the gap; does not exceed greedy.
- C2d (Pareto improvement): **falsified** — no regime where v4 > greedy.
