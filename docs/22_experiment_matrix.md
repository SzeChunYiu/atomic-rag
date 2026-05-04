# Experiment Matrix

Initial systems:

A0: BM25 top-k
A1: dense top-k
A2: hybrid top-k
A3: dense + reranker
B1: dense + evidence-SNR rerank
B2: hybrid + evidence-SNR rerank
C1: dense + sparse selector
C2: dense + SNR + sparse selector
D1: SNR + sparse selector + reranker

Each system must run with same dataset split and budget.
