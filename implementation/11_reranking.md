# Reranking Module

Purpose:
Support strong baselines and optional final ranking.

Reranker types:
- cross-encoder
- late interaction if available
- LLM relevance judge only for analysis, not default benchmark

Rules:
- log reranker calls
- log latency
- log cost if API-based
- compare with and without reranker

Reranking must not hide detector failures.
