# IRC-robustness empirical results

## Chunk-size sweep (collinear safety, recall@1 vs chunk_size)

| selector | mean | stdev | range | values |
|---|---|---|---|---|
| anti_kt | 1.000 | 0.000 | 0.000 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| greedy | 1.000 | 0.000 | 0.000 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| mmr | 1.000 | 0.000 | 0.000 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

**Theorem-1 chunk-perturbation pass condition (anti_kt.stdev < min(other.stdev)): `FAIL`**

## Distractor-injection sweep (infrared safety, recall@1 vs n_distractors)

| selector | mean | stdev | range | values |
|---|---|---|---|---|
| anti_kt | 0.800 | 0.267 | 0.667 | [1.0, 1.0, 1.0, 0.6666666666666666, 0.3333333333333333] |
| greedy | 0.800 | 0.267 | 0.667 | [1.0, 1.0, 1.0, 0.6666666666666666, 0.3333333333333333] |
| mmr | 0.800 | 0.267 | 0.667 | [1.0, 1.0, 1.0, 0.6666666666666666, 0.3333333333333333] |

**Theorem-1 IR pass condition (anti_kt.stdev < min(other.stdev)): `FAIL`**
