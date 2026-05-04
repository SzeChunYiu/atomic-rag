# Experiment: Sparse Selection

Goal:
Test whether fewer selected atoms preserve answer quality.

Run:
- use top-30 candidates
- select under budgets: 512, 1024, 2048 tokens
- compare against top-k chunks with same budget

Measure:
- answer F1 or correctness
- faithfulness
- citation accuracy
- token count
- latency

Sanity:
Verify no required query facet is dropped.
