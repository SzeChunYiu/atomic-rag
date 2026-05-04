# Baselines

Required baselines:
- BM25
- dense retrieval
- hybrid BM25 + dense
- dense + cross-encoder reranker
- ColBERT-style late interaction if feasible
- hierarchical retrieval such as RAPTOR if feasible
- sparse/context selection baseline
- long-context top-k baseline

Rule:
Start simple, then add stronger baselines.
Do not publish claims against weak baselines only.
