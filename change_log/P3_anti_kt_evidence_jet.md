# P3 — Anti-$k_T$ evidence-jet clustering (the first novel mechanism)

**Goal.** Ship the first novel selector and *prove* (theoretically + empirically)
its IRC-safety properties: that the leading evidence jet is invariant under
chunk-boundary splits and low-relevance distractor injections.

## Mechanism statement

Existing selectors — MMR, DPP, submodular greedy — produce a context that
depends sensitively on chunk size and on distractor pool composition. Empirically
this is one of the brittle places in modern RAG: change chunker, change result.
Anti-$k_T$ borrows from particle physics a clustering operator with two
provable invariances that map exactly onto the brittlenesses RAG suffers:

- **Collinear safety**: splitting an evidence-bearing chunk into halves does not
  change the leading jet's hard support set. Targets F4 (split evidence).
- **Infrared safety**: adding arbitrarily many low-relevance distractor atoms
  cannot dislodge any hard atom from the leading jet. Targets F2 / F5.

A retrieval pipeline whose selector inherits these invariances is, by
construction, robust to two of the most common perturbations in benchmark
practice. Whether this robustness *also* yields better-than-baseline absolute
performance is an empirical question we will answer in the chunk-size and
distractor-pool sweeps.

## Done

- `selection/anti_kt.py` — full implementation of the clustering algorithm
  with sequential recombination, beam emission, and history trace (175 lines).
- `selection/jet_select.py` — pipeline adapter consuming `EvidenceAtom` +
  chunk embeddings and producing `SelectedRecord`.
- `selection/mmr.py` — Maximal Marginal Relevance baseline (Carbonell &
  Goldstein 1998), λ-tunable.
- `pipeline/select_run.py` — selector mode switch (greedy | anti_kt | mmr),
  loads chunk embeddings on demand for the diversity-aware selectors.
- `methods/08_anti_kt_irc_safety.md` — IRC-safety theorem statement (collinear
  + infrared), proof sketch, falsification protocol.
- `tests/test_anti_kt.py` — algorithm correctness, collinear-split invariance,
  infrared hard-atom preservation, $R$-dependence, empty-input edge case.
- `tests/test_jet_select_pipeline.py` — end-to-end with anti-$k_T$ and MMR.
- Test suite **48/48** pass.

## Planned (P3 finalization on real data)

- IRC-safety empirical test (the headline figure):
  chunk-size sweep $\{256, 384, 512, 640, 768\}$ × selector $\{$greedy, MMR, anti-$k_T \}$
  on HotpotQA-1k. Pass condition: anti-$k_T$ F1 variance < MMR F1 variance
  with paired bootstrap $p < 0.01$.
- Distractor-pool sweep: inject $\{0, 5, 10, 25, 50\}$ low-relevance atoms;
  same pass condition.
- Push to LUNARC, queue both sweeps, pull artifacts, plot.

## Not done (queued for P3.5)

These are bonus methods *added* this turn (per user request) and slot between
the anti-$k_T$ ship and P4:

- CLEAN-RAG iterative residual selector.
- FDR (Benjamini-Hochberg) candidate gate.
- Aperture-photometry SNR (cosine-ball local background).
- Coronagraphy (anchor-mask re-retrieval).
- VLBI cross-correlation of independent embedders.
- Cherenkov-threshold candidate cut.
- Standard-candle absolute-relevance calibration.

## Reproduction

```bash
python -m pytest tests/test_anti_kt.py tests/test_jet_select_pipeline.py -q

# Run anti-kT selector via the standard benchmark CLI:
cat > /tmp/anti_kt.yaml <<EOF
dataset: tiny_anti_kt
seed: 0
paths:
  corpus_path: data/tiny/corpus.jsonl
  queries_path: data/tiny/queries.jsonl
  gold_path: data/tiny/gold.jsonl
  output_dir: runs/tiny_anti_kt
chunk_size: 120
chunk_overlap: 20
embedding: {use_hash_embedder: true}
retriever: {candidate_top_n: 5, mode: fusion_rrf}
detector: {window: 3}
selector: {token_budget: 256, mode: anti_kt, anti_kt_R: 1.0, anti_kt_n_jets: 1}
generator: {enabled: true, provider: stub}
metrics: {ks: [1, 3]}
EOF
rag-run-benchmark --config /tmp/anti_kt.yaml
```
