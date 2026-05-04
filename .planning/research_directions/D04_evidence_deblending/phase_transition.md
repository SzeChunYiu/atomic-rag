# Phase Transition Analysis

## Claim
Multi-hop RAG has a *detectability threshold* C\*: as evidence crowding rises, P(support_chain_complete) drops sharply rather than smoothly. Different retrieval/selection systems shift C\* but do not eliminate the threshold.

## What we measure

For each system S and each crowding axis x (e.g. `n_distractors_per_gold`):

```
P_S(success | x) = mean over queries of support_chain_complete
C*_S(x) = smallest x such that P_S(success | x) < 0.5
AUC_S(x) = area under P_S(success | x) curve, integrated over x
```

We also fit a logistic to `P_S(success | x)` and report slope at C\*. A sharp slope is evidence of a transition; a shallow slope is evidence of smooth degradation.

## Headline table

| System | C\*_dense | C\*_typed | C\*_entity | AUC | p95 latency | tokens |
|--------|----------|----------|-----------|-----|-------------|--------|
| BM25 chunk | | | | | | |
| Dense chunk | | | | | | |
| Hybrid + reranker | | | | | | |
| Atom dense (raw) | | | | | | |
| Atom + local-bg detector | | | | | | |
| Atom + graph reconstr. | | | | | | |

Filled in only after the synthetic benchmark runs.

## Why this matters for the paper

Two outcomes both produce a paper:
1. **Threshold exists, our method moves it.** The deblending detector pushes C\* further into denser regimes than chunk baselines at fixed token budget. This is the strong-method result.
2. **Threshold exists, our method does not move it.** Then the contribution is the diagnostic: prior systems all fail in the same regime, evidence crowding is a structural failure mode of chunk-and-cosine RAG. This is the strong-benchmark result.

The losing outcome — no threshold and our method doesn't help — kills the direction. We accept that.

## Falsification rules (matches brief §10)

- Rule A: `all_gold_atoms_selected_rate` improves ≥4pp over chunk baseline; or HotpotQA `all_gold_in_sel` rises ~0.858 → >0.90.
- Rule B: local-background detectability beats raw atom cosine on support-chain recall.
- Rule C: success curves show measurable degradation as crowding increases AND our method shifts the threshold.
- Rule D: long-context top-k must not trivially beat us at acceptable token cost.
- Rule E: strong baselines (ColBERTv2, SPLADE, HippoRAG) must be beaten or contribution shifts to diagnostic.

## Implementation
- `src/astro_cs_rag/diagnostics/phase_transition.py` — fit P(success | x), estimate C\*, compute AUC, logistic slope.
- `src/astro_cs_rag/reporting/phase_reports.py` — render the headline table.
