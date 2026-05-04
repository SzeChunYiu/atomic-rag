from astro_cs_rag.atoms.schemas import Query
from astro_cs_rag.evaluation.metrics import evaluate_ranked_queries


def test_recall_at_one_doc_level() -> None:
    queries = [
        Query(query_id="q1", text="x", gold_doc_ids=["a"]),
    ]
    rankings = {"q1": ["c1", "c2"]}
    c2d = {"c1": "b", "c2": "a"}
    m = evaluate_ranked_queries(queries, rankings, c2d, ks=[1, 2])
    assert m["recall@1_doc_mean"] == 0.0
    assert m["recall@2_doc_mean"] == 1.0

