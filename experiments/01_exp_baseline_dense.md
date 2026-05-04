# Experiment: Dense Baseline

Goal:
Establish a simple dense retrieval baseline.

Run:
- dataset subset
- fixed embedding model
- top-k = 5, 10, 20, 100

Measure:
- Recall@k
- NDCG
- latency
- tokens if sent to generator

Artifacts:
- candidates.jsonl
- metrics.json
- report.md

This is the baseline Astro-RAG must improve.
