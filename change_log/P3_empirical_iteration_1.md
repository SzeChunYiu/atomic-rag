# P3 empirical — iteration 1 (anti-$k_T$ IRC robustness, synthetic stress)

## What ran

1. **Tiny corpus IRC sweep** — chunk_sizes ∈ {60, 90, 120, 150, 180, 240}, selectors ∈ {greedy, mmr, anti_kt}.
2. **Synthetic corpus** with deliberate two-sentence gold pairs and disjoint-vocabulary distractors, same sweep.
3. **Bootstrap (1000 resamples)** on the synthetic results' chunk-stdev statistic.

## Key findings (mechanism statement: positive but underpowered)

### Bottleneck identified
Hash embedder retrieval caps recall@1 at ~10% on synthetic. The selection layer is *measurable* only on a metric that is sensitive to selection per se, not to retrieval. Using `gold_pair_coverage` (fraction of queries whose selected_context contains the gold's joint two-sentence answer span) decouples the two.

### Pareto domination of anti-$k_T$ vs greedy on the right metric

| selector | gold_pair_coverage mean | stdev | CV |
|---|---|---|---|
| **anti_kt** | **0.694** | **0.099** | **0.142** |
| greedy | 0.661 | 0.121 | 0.183 |
| mmr | 0.028 | 0.023 | 0.825 |

Anti-$k_T$ has both higher mean **and** lower stdev than greedy → strict Pareto domination. CV is 22% lower → IRC stability claim has empirical support.

### MMR collapse — diagnostic
MMR is an *anti-baseline* on this benchmark: it explicitly de-prioritizes redundant chunks, but the gold pair is two halves of the *same* doc and looks redundant under cosine similarity. MMR's near-zero coverage with low variance is failure-with-low-variance, not robustness. The Pareto-aware pass condition correctly identifies this.

### Bootstrap (n=1000 resamples on chunk-size axis)

```
anti_kt stdev: 0.085 (90% CI [0.025, 0.127])
greedy  stdev: 0.106 (90% CI [0.062, 0.154])
mmr     stdev: 0.020 (90% CI [0.012, 0.027])

P(anti_kt.stdev < greedy.stdev) = 0.759
```

**The directional result is real but underpowered for $p < 0.05$ at this experiment scale.**

## Diagnosis

| Issue | Cause | Fix queued |
|---|---|---|
| recall@1 capped at 10% | hash embedder produces nearly-random vectors | use BGE-M3 in next iteration |
| bootstrap probability only 76% | only 6 chunk sizes × 30 queries; sample too small | expand to 20 chunk sizes × 200 queries × 5 seeds |
| MMR's "low stdev" is failure | gold pair is redundant in MMR's cosine similarity | accept; MMR is the wrong selector for joint-evidence problems — note in paper |

## What this iteration *does* deliver

- A **selection-sensitive metric** (`gold_pair_coverage`) that genuinely separates selectors.
- A **synthetic benchmark family** controllable in chunk size, distractor count, and topic disjointness.
- A **Pareto-aware pass condition** for IRC-safety claims that doesn't reward MMR's failure-with-low-variance.
- The **first directional empirical signal** that anti-$k_T$ is more chunk-stable than greedy.

## What's next (iteration 2)

- **Larger synthetic experiment** (200 queries × 20 chunk sizes × 5 seeds) to lift bootstrap power above $p < 0.01$.
- **Real BGE-M3 retrieval**, lifting recall@1 above the noise floor and exposing `recall@1`-axis differences too.
- **Distractor-injection sweep** (IR safety): vary $N \in \{0, 5, 10, 25, 50, 100\}$ and check that anti-$k_T$'s leading-jet hard support set is preserved.
- **LUNARC Batch 1** (already running) gives the real-data complement.

## Files written this iteration

- `scripts/synthetic_irc_experiment.py` — synthetic corpus + sweep
- `scripts/bootstrap_irc.py` — 1000-resample bootstrap test
- `runs/synthetic_irc/synthetic_irc.csv` — raw results
- `runs/synthetic_irc/results.md` — markdown summary
- `runs/synthetic_irc/bootstrap.json` — bootstrap distribution

## Reproduction

```bash
python scripts/synthetic_irc_experiment.py
python scripts/bootstrap_irc.py
cat runs/synthetic_irc/results.md
```
