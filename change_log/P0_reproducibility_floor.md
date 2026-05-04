# P0 — Reproducibility floor

**Goal:** before any novel method, lock down a runnable end-to-end pipeline
(retrieve → detect → select → **generate** → evaluate) on real data with full
artifact logging and reproducible Ollama generation.

## Mechanism statement (P0 contribution to understanding RAG)

P0 contributes nothing novel; it is the measurement instrument. The point is
that all subsequent claims are anchored to a calibrated baseline whose
end-to-end answer accuracy can be decomposed into stage-level efficiencies
($\varepsilon_\text{retrieval}, \varepsilon_\text{select}, \varepsilon_\text{generate}$).

## Done

- Audit of Cursor scaffold (kept verbatim — schemas, configs, indexing, BM25,
  dense, RRF fusion, SNR detection, greedy selection, ranking metrics, line-limit CI).
- `AGENTS.md` updated with phased program and atomic discipline.
- `change_log/` folder + atomic failure atlas (F1–F9).
- HuggingFace dataset preparer for HotpotQA-distractor + NQ-open subsets
  (`src/astro_cs_rag/data/hf_loaders.py` + `cli/prepare_data.py`).
- BGE-M3 default for `SentenceEmbedder`; embedding model digest logged in `index_meta.json`.
- Cross-encoder reranker baseline (`reranking/cross_encoder.py`,
  `pipeline/rerank_run.py`, `cli/rerank.py`).
- Ollama generation client + generator + pipeline stage
  (`generation/ollama_client.py`, `generation/generator.py`,
  `pipeline/generate_run.py`, `cli/generate.py`).
- Answer-quality metrics (EM, token F1, citation accuracy)
  (`evaluation/answer_metrics.py`).
- Pipeline integration: `benchmark_run` now optionally reranks and generates,
  and `evaluate_run` reports answer metrics if answers were generated.
- New tests: `test_hf_loaders.py`, `test_answer_metrics.py`,
  `test_generator_stub.py`, `test_end_to_end_tiny.py`.
- Smoke run on `data/tiny/` with stub generator passes.

## Planned (P0 finalization on real data)

- Run BM25, dense (BGE-M3), hybrid, dense+rerank baselines on HotpotQA-1k and
  NQ-open-1k with Llama-3.1-8B-Instruct (Ollama). Verify recall@10 within ±2 pp
  of published numbers for the same model + same subset.
- Lock model digests (`ollama show llama3.1:8b-instruct-q4_K_M --modelfile`)
  and embedder revision into `runs/<id>/manifest.json`.
- Write the reproducibility receipt (one command, byte-stable artifacts).

## Not done (deferred to later phases)

- ColBERT, RAPTOR, SPLADE++ baselines → P2.
- Asimov benchmark → P2.
- Atomic decomposer (sentences → claim atoms with entities/numbers) → P1.
- Calorimetric query archetype profiler → P1.
- Real faithfulness scorer (NLI cross-encoder + conservation-law residuals) → P5.
- LUNARC scale runs → P7.

## Reproduction (after P0 land)

```bash
pip install -e ".[dev,hf]"
ollama pull llama3.1:8b-instruct-q4_K_M
rag-prepare-data --dataset hotpotqa --split validation --n 1000 \
  --out data/hotpotqa_1k
rag-run-benchmark --config configs/benchmarks/hotpotqa_1k_dense_rerank.yaml
```
