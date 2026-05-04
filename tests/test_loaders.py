import json
from pathlib import Path

from astro_cs_rag.data.loaders import load_corpus_jsonl, load_queries_jsonl


def test_load_tiny_corpus_and_queries() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = load_corpus_jsonl(root / "data/tiny/corpus.jsonl")
    qs = load_queries_jsonl(root / "data/tiny/queries.jsonl")
    assert len(docs) == 5
    assert len(qs) == 3
    assert docs[0].doc_id == "doc_alpha"
    assert qs[0].query_id == "q1"
