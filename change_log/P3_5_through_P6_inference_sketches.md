# P3.5 – P6 — Bonus methods + inference sketches (shipped)

This change_log batches the post-anti-$k_T$ wave of methods. Each module is
small, tested, and ready for ablation runs on real data.

## Mechanism statement (cumulative)

The pipeline now has a **complete F1–F9 coverage matrix**. For every atomic
failure mode there is at least one implemented method addressing it
(see `methods/09_layered_pipeline_architecture.md`). The remaining work is
empirical: run all baselines + novel methods on real benchmarks, fill the
per-archetype performance table, and identify the *minimum sufficient* mix.

## Done

### Layer-3 (multi-channel score combination)
- `retrieval/lockin.py` — coherent / incoherent paraphrase aggregation.
  Provides per-chunk *coherence ratio* as a learned-noise diagnostic.
- `retrieval/vlbi.py` — VLBI-style cross-correlation of independent score
  fields with fringe-visibility quality control.
- `generation/paraphrase.py` — deterministic paraphrase generator (Ollama).

### Layer-4 (background subtraction / SNR)
- `detection/aperture.py` — cosine-ball aperture-photometry SNR.
- `retrieval/coronagraph.py` — anchor-mask residual retrieval.

### Layer-5 (selection threshold)
- `detection/cherenkov.py` — calibrated relevance threshold (median+MAD,
  quantile, fixed).
- `detection/fdr.py` — Benjamini–Hochberg FDR control on z-scores.
- `detection/standard_candle.py` — isotonic absolute-relevance calibration.

### Layer-6 (sparse evidence reconstruction)
- `selection/clean_rag.py` — Högbom CLEAN iterative residual selector.
- `selection/sbi.py` — simulation-based-inference posterior over evidence
  subsets (simulator + featurizer + stand-in posterior; full neural ratio
  estimator deferred to P7).
- `selection/smc.py` — sequential-Monte-Carlo / particle-filter adaptive
  selector with effective-sample-size-triggered resampling.

### Layer-1 (query reformulation) and inference primitives
- `detection/unfolding.py` — graph-Laplacian regularized unfolding of
  embedding smearing (lightweight stand-in for diffusion priors).
- `retrieval/wasserstein.py` — Sinkhorn entropy-regularized OT between
  query and chunk token measures.
- `diagnostics/tda.py` — H0 persistence (union-find sweep) + persistence
  entropy / mean / max for the per-query candidate subgraph.
- `diagnostics/ood_gate.py` — Gaussian OOD detector calibrated at α=0.05;
  stand-in for full normalizing-flow CATHODE-style detector.

### Layer-8 (faithfulness)
- `evaluation/conservation.py` — entity / numeric / temporal conservation
  residuals with bounded mean residual + faithfulness score; integrated
  into `pipeline/evaluate_run.py`. Writes `conservation_residuals.jsonl`
  alongside metrics when generated answers are present.

## Tests

- `tests/test_anti_kt.py` — algorithm + IRC-safety (5 tests).
- `tests/test_jet_select_pipeline.py` — anti-$k_T$ + MMR end-to-end.
- `tests/test_p3_5_modules.py` — aperture, FDR, Cherenkov, coronagraph,
  VLBI, standard-candle (9 tests).
- `tests/test_lockin.py` — coherent/incoherent sum, phase patterns (4 tests).
- `tests/test_conservation.py` — entity/numeric/temporal residuals (6 tests).
- `tests/test_p5_p6_modules.py` — Sinkhorn-OT, TDA H0 persistence, OOD,
  unfolding, SBI featurizer, SMC (10 tests).

Full suite: **77 / 77 pass**.

## Reproduction

```bash
# Anti-kT selector via the standard benchmark CLI:
rag-run-benchmark --config configs/benchmarks/tiny_stub.yaml  # default greedy

# Selector mode is hot-swappable:
sed -i.bak 's/mode: greedy/mode: anti_kt/' configs/benchmarks/tiny_stub.yaml
rag-run-benchmark --config configs/benchmarks/tiny_stub.yaml

# Lockin retrieval (paraphrase requires Ollama; uses original query if disabled).
sed -i.bak 's/mode: fusion_rrf/mode: lockin/' configs/benchmarks/tiny_stub.yaml
rag-run-benchmark --config configs/benchmarks/tiny_stub.yaml
```

## Not done (explicit deferrals)

- Full neural ratio estimator for SBI (training pipeline + dataset of
  simulated triples) → **P7** under LUNARC compute.
- Full diffusion-prior unfolding (a learned reverse SDE replacing the
  Gaussian-Laplacian solver) → **P7** with A100 training run.
- Real CATHODE/ANODE normalizing-flow OOD detector → **P7**, requires
  training a flow on millions of query embeddings.
- ColBERT-v2 PLAID via pylate (vs in-house BGE-M3 multivec MaxSim) → **P7**.
- Empirical IRC-safety chunk-perturbation curve on HotpotQA-1k → next turn.
- Empirical per-archetype recall table → next turn.
