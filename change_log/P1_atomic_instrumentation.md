# P1 — Atomic instrumentation + literature reconnaissance

**Goal.** Make every atomic failure mode *visible* via measurable artifacts
before any novel method is introduced. Build the instruments. Identify the
prior art that determines our novelty surface.

## Mechanism statement

A RAG pipeline is a measurement chain whose stages — encoder, similarity,
threshold, chunker, context assembly, generator — each lose information.
Before designing operators that recover information, we need to *measure*
where information is lost. P1 supplies two instruments: the **claim-atom
decomposer** (resolves chunks into evidential units) and the **calorimetric
score-shape profiler** (resolves the morphology of the per-query relevance
field). With these, the per-archetype baseline gap becomes inspectable.

## Done

- `atoms/decomposer.py` — chunk → sentence → ClaimAtom with entity/number/date
  tagging via regex (spaCy-free for CI portability).
- `atoms/schemas.py` — added `ClaimAtom` typed record.
- `pipeline/decompose_run.py` + `cli/decompose.py` — materializes
  `claim_atoms.jsonl` next to a built index, with per-chunk yield manifest.
- `diagnostics/calorimetry.py` — score-field shape statistics: peak height,
  peak-minus-median, FWHM-index fraction, kurtosis, skewness, SAS bimodality
  coefficient, second-peak gap, log-rank tail-decay slope; archetype label
  ∈ {compact, plateau, bimodal, diffuse, noisy, empty}.
- `pipeline/profile_run.py` + `cli/profile.py` — per-query shape rows
  (`query_shapes.jsonl`) and aggregate (`archetype_summary.json`) with
  per-archetype recall@k means when a gold file is supplied.
- `change_log/atomic_failure_atlas.md` — F1–F9 taxonomy + method-to-failure
  assignment matrix + per-archetype hypothesized failure mapping.
- `research/literature_review/recon/` — six reconnaissance memos
  (template + anti-$k_T$ + lock-in + unfolding + conservation + calorimetric
  routing + Asimov benchmark).
- Tests — `test_decomposer.py`, `test_calorimetry.py`, `test_profile_run.py`;
  full suite 28/28 pass.

## Planned (P1 finalization on real data)

- Run `rag-prepare-data --dataset hotpotqa --n 1000` then profile the
  dense+rerank baseline; fill the empirical recall-per-archetype table in
  `atomic_failure_atlas.md`.
- Same for NQ-open-1k.
- Cross-reference: do the predicted archetype-to-failure assignments hold?
- Complete the prior-art search for each recon memo (target: each row of the
  "closest prior art" table backed by a real citation, not a placeholder).

## Not done (deferred)

- spaCy-backed decomposer for higher-quality entity / dependency extraction
  → P3 if regex precision blocks anti-$k_T$ clustering.
- Multi-hop bridge-entity tagger → P3.
- Asimov benchmark generator → P2.
- Cross-section metric formalism → P2.

## Reproduction

```bash
# Profile the existing tiny smoke run.
rag-profile --run-dir $(ls -d runs/tiny_stub/* | grep -v index_bundle | tail -1) \
  --gold-path data/tiny/gold.jsonl --k 1 --k 3
# Decompose a built index into claim atoms.
rag-decompose --index-dir runs/tiny_stub/index_bundle
```

## What P1 teaches us about RAG (mechanism statement, recap)

The score field's morphology is itself a signal — independent of any retriever
or model. Different morphologies imply different dominant failure modes, and
no single retrieval recipe is optimal across them. This is the first
empirical justification for **archetype-conditional retrieval** — the
calorimetric routing program of P5+.

The claim-atom decomposition is the substrate on which P3+ methods (anti-$k_T$
clustering, conservation residuals, lock-in coherent rerank) become
expressible at all. Without it, we have no atomic addressability.
