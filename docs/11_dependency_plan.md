# Dependency Plan

Core Python stack:
- Python 3.11+
- pydantic for schemas
- typer or argparse for CLIs
- numpy and scipy for scoring
- pandas for result tables
- scikit-learn for baseline tools
- faiss or hnswlib for ANN experiments
- sentence-transformers for embeddings
- pyserini or rank-bm25 for BM25
- pytest for tests

Keep dependencies modular.
Do not force expensive libraries into small experiments.
