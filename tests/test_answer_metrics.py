from astro_cs_rag.evaluation.answer_metrics import (
    citation_accuracy,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_squad_style() -> None:
    assert normalize_answer("The Crab Nebula!") == "crab nebula"


def test_em_handles_articles_and_punct() -> None:
    assert exact_match("The Crab Nebula.", ["crab nebula"]) == 1.0
    assert exact_match("Crab", ["crab nebula"]) == 0.0


def test_f1_partial() -> None:
    f1 = token_f1("Crab Nebula pulsar", ["crab nebula"])
    assert 0.0 < f1 < 1.0


def test_citation_accuracy() -> None:
    chunk_to_doc = {"c1": "doc_a", "c2": "doc_b", "c3": "doc_c"}
    assert citation_accuracy(["c1", "c2"], chunk_to_doc, ["doc_a"]) == 0.5
    assert citation_accuracy([], chunk_to_doc, ["doc_a"]) == 0.0
    assert citation_accuracy(["c1"], chunk_to_doc, []) == 0.0
