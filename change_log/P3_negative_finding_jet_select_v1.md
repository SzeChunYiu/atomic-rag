# P3 — negative empirical finding on real HotpotQA, anti-$k_T$ v1

## What ran

LUNARC Batch 1 (job 2997181, A100 80GB, hotpotqa-1k):
- `dense + greedy` ✓
- `dense + anti_kt` ✓
- `dense + mmr` (running)
- `fusion_rrf + {greedy, anti_kt, mmr}` (queued; likely TIMEOUT)

## Headline result

| metric | greedy | anti_kt (v1) | delta |
|---|---|---|---|
| recall@1 (doc) | 0.443 | 0.443 | 0 (shared retrieval) |
| recall@5 (doc) | 0.842 | 0.842 | 0 |
| **citation_accuracy** | **0.886** | **0.014** | **−98.4%** |
| **answer_F1** | **0.097** | **0.007** | **−92.8%** |
| conservation_faithfulness | 0.697 | 0.705 | +0.008 |

**Anti-$k_T$ v1 catastrophically underperforms greedy on HotpotQA.** Citation
accuracy collapses from 88.6% → 1.4%.

## Atomic diagnosis (CLAUDE.md framework)

1. **Which atom changed?** The selector function. Both runs share retriever
   output — recall@k is identical. The difference is which K chunks make it
   into the budget.
2. **Which metric moved?** Citation accuracy and answer F1, both at the
   chunk-content level.
3. **Which failure mode was fixed?** None on this benchmark.
4. **Which new failure mode appeared?** **F-jet-exclusion** — when
   `n_jets=1`, only the leading jet's members are eligible for the budget.
   On multi-hop QA where the gold spans heterogeneous topics, the leading
   jet by summed relevance can be a cluster of high-score-but-redundant
   distractors; the gold chunk lives in jet #2 or #3 and is silently
   excluded.
5. **Real or noise?** Real, ~99% relative drop on n=1000 queries — far
   beyond bootstrap noise.

## Root cause (mechanism)

In `selection/jet_select.py`, the original packing logic was:

```python
for jet in result.final_jets[:n_jets]:   # n_jets=1 by default
    for cid in jet.member_atom_ids:
        ranked_chunks.append((cid, jet.relevance))
```

With `n_jets=1`, all budget tokens go to the leading-jet members, by
internal merge order — chunk SNR is *not* a tiebreaker. On HotpotQA's
heterogeneous candidate pools, the leading jet is often a tight cluster of
near-duplicate distractors that look mutually similar in the embedder
space; the actual gold passages are off-topic enough to land in *separate*
jets and are excluded.

## Fix (anti-$k_T$ v2)

Two changes in `selection/jet_select.py` + `config/schema.py`:

1. Default `anti_kt_n_jets` from `1` → `-1` (sentinel: pack across **all**
   jets in jet-relevance order until the budget is full).
2. Within each jet, order members by chunk SNR (descending) before packing.

Physics intuition: in collider analyses you don't only inspect the leading
jet — you study the leading 2–6 jets and use their kinematics. Restricting
ourselves to jet #1 was an arbitrary, non-IRC-safe choice we didn't justify.
The clustering decision is still IRC-safe; we just no longer use it as a
hard exclusion.

## What this preserves (IRC theorem still holds)

- Clustering remains anti-$k_T$. Adding low-relevance chunks does not
  change which jet wins.
- Splitting a chunk into halves yields the same merged jet.
- Packing is now a soft preference (jet relevance) + chunk SNR tiebreaker,
  not a hard exclusion of non-leading jets.

## Why iter 1 (synthetic) showed the *opposite*

In the synthetic IRC test, the gold pair was *constructed to cluster*: both
halves shared the topic word and the same surrounding text. The leading
jet *was* the gold pair. Greedy missed one half because the second half had
slightly lower retrieval score. There, anti-$k_T$ v1's leading-jet
preference was the right inductive bias.

On real HotpotQA, gold spans cross documents about different topics
(e.g., "country of birth of person who founded the company that makes X");
they cluster *less* tightly than near-duplicate distractors. The leading
jet excludes them.

The fix (v2) should preserve the synthetic win and recover greedy
performance on HotpotQA.

## What's next

1. Re-run synthetic IRC iter 2 under anti-$k_T$ v2 → confirm Pareto
   domination of greedy is preserved (or stronger, since lower-jet members
   now contribute).
2. Resubmit `hotpotqa_1k__dense__anti_kt` with v2 → expect citation
   accuracy ≈ greedy's 0.886.
3. If v2 ≥ greedy on HotpotQA AND v2 ≥ greedy on synthetic, the IRC-safety
   claim is supported on real data.
4. If v2 ≈ greedy on HotpotQA, the IRC-safety advantage is mechanism-
   real but task-bounded — the paper claim becomes "anti-$k_T$ is a safe
   replacement for greedy with stability guarantees, equal accuracy, and
   demonstrably tighter behavior under chunk-size perturbation."

## Files changed

- `src/astro_cs_rag/selection/jet_select.py` — pack across all jets,
  member SNR ordering.
- `src/astro_cs_rag/config/schema.py` — default `anti_kt_n_jets = -1`.

## Operational note (deployment bug)

The first v2 run on LUNARC (job 2997695) returned **citation_accuracy = 0**
— even worse than v1's 0.014. Diagnosis revealed `astro-cs-rag` was
installed with `pip install .` (frozen copy in `site-packages/`), not
`pip install -e .`. So the v2 source edits were never picked up by the
benchmark. Worse, the slurm script forced `anti_kt_n_jets = -1` in the
config; the *old* code in site-packages then evaluated
`final_jets[:-1]` = "all jets except the last" — empty list when there
is one jet → nothing selected.

Fix: `pip install -e .` on the LUNARC venv (job re-submitted as 2997929).
This is now part of the deployment checklist; future code changes flow
into running benchmarks immediately via rsync without reinstall.

The v1 numbers above (citation_accuracy 0.014, answer_F1 0.007) are the
correct v1 baseline. The v2 numbers from job 2997695 are NOT a true v2
test — they are the v1 code with `n_jets=-1` which silently produced an
empty selection. Treat job 2997929 (post-editable-install rerun) as the
authoritative v2 result.
