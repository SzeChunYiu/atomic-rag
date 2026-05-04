# Synthetic IRC empirical results

## recall@1 vs chunk_size

| selector | mean | stdev | CV | values |
|---|---|---|---|---|
| anti_kt | 0.100 | 0.051 | 0.509 | [0.1, 0.167, 0.167, 0.033, 0.067, 0.067] |
| greedy | 0.100 | 0.051 | 0.509 | [0.1, 0.167, 0.167, 0.033, 0.067, 0.067] |
| mmr | 0.100 | 0.051 | 0.509 | [0.1, 0.167, 0.167, 0.033, 0.067, 0.067] |

**Pareto-aware pass on `recall@1`: `FAIL` (anti_kt on the (mean, −stdev) frontier vs baselines)**

## answer_f1 vs chunk_size

| selector | mean | stdev | CV | values |
|---|---|---|---|---|
| anti_kt | 0.028 | 0.031 | 1.096 | [0.058, 0.078, 0.034, 0.0, 0.0, 0.0] |
| greedy | 0.017 | 0.018 | 1.073 | [0.011, 0.055, 0.02, 0.0, 0.007, 0.007] |
| mmr | 0.006 | 0.010 | 1.830 | [0.006, 0.028, 0.0, 0.0, 0.0, 0.0] |

**Pareto-aware pass on `answer_f1`: `FAIL` (anti_kt on the (mean, −stdev) frontier vs baselines)**

## gold_pair_coverage vs chunk_size

| selector | mean | stdev | CV | values |
|---|---|---|---|---|
| anti_kt | 0.694 | 0.099 | 0.142 | [0.533, 0.867, 0.7, 0.733, 0.667, 0.667] |
| greedy | 0.661 | 0.121 | 0.183 | [0.533, 0.867, 0.667, 0.5, 0.7, 0.7] |
| mmr | 0.028 | 0.023 | 0.825 | [0.033, 0.067, 0.0, 0.0, 0.033, 0.033] |

**Pareto-aware pass on `gold_pair_coverage`: `PASS` (anti_kt on the (mean, −stdev) frontier vs baselines)**

