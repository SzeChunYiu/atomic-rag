# P0 reproducibility receipt

The working pipeline as of 2026-05-02. Use this as the trust anchor for any
future change.

## Smoke run (no network, no GPU)

```bash
pip install -e ".[dev]"
pytest -q                                                    # 19/19 pass
rag-run-benchmark --config configs/benchmarks/tiny_stub.yaml # writes runs/tiny_stub/<id>/
```

Expected smoke-run artifacts under `runs/tiny_stub/<id>/`:

```
config.yaml             — frozen config snapshot
manifest.json           — corpus/query/chunk counts + paths
candidates.jsonl        — typed Candidate rows with retriever provenance
scores.jsonl            — per-query bm25/dense/fusion top-K diagnostics
evidence_atoms.jsonl    — typed EvidenceAtom rows with SNR + background
selected_context.jsonl  — chunks chosen under token budget
dropped_candidates.jsonl — what was discarded and why
coverage_trace.jsonl    — per-decision selection trace
generated_answers.jsonl — typed GeneratedAnswer rows with citation resolution
generation_meta.json    — provider, model, ollama show payload, latency
metrics.json            — retrieval (recall/precision/NDCG/MRR) + answer (EM/F1/cite-acc)
timing.json             — retrieve + evaluate seconds
report.md               — deterministic markdown summary
run_meta.json           — start time + run id
```

Smoke-run numbers (tiny corpus, hash embedder, stub generator):

| metric | value |
|---|---|
| recall@1_doc_mean | 1.0 |
| ndcg@3_chunk_mean | 1.0 |
| mean_reciprocal_rank_doc | 1.0 |
| citation_accuracy_mean | 1.0 |
| answer_em_mean | 0.0 (tiny data has no gold answer text) |

## Real-data run (requires network + Ollama)

```bash
pip install -e ".[dev,hf]"
ollama pull llama3.1:8b-instruct-q4_K_M
rag-prepare-data --dataset hotpotqa --split validation --n 1000 \
    --out data/hotpotqa_1k
rag-run-benchmark --config configs/benchmarks/hotpotqa_1k_dense_rerank.yaml
```

Reports land under `runs/hotpotqa_1k_dense_rerank/<run_id>/`. The same command
re-run with the same seed must produce byte-identical artifacts (modulo
timestamps in `run_meta.json` and `timing.json`).

## What is logged for reproducibility

- Embedder revision digest (`index_meta.embedding_digest.revision`).
- Reranker model name in `rerank_meta.json`.
- Ollama model card via `/api/show` in `generation_meta.json.ollama_show`.
- Prompt style + temperature + seed in `config.yaml`.
- Full input contract in `manifest.json` (corpus path, query path, counts).

## Known limitations of P0 (intentional)

- BGE-M3 used as a single-vector encoder (no multi-vector retrieval yet).
- Rerank uses cross-encoder logits as scalar; no calibrated probabilities yet.
- Faithfulness metric is citation-accuracy proxy only; entailment + conservation
  residuals come in P5.
- No paid-API generator path; add later if needed for reviewer comfort.
