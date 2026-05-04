"""Unit tests for the gold-atom audit failure-layer classifier."""

from __future__ import annotations

from astro_cs_rag.diagnostics.gold_atom_audit import (
    AtomRecord,
    GoldRecord,
    QueryArtifacts,
    aggregate,
    audit_query,
)


def _atoms():
    return [
        AtomRecord("a1", "c1", "d1", "Mara Keene was born in Norway.", "WHERE"),
        AtomRecord("a2", "c1", "d1", "She directed Liora's Map.", "WHO"),
        AtomRecord("a3", "c2", "d2", "Oslo is the capital of Norway.", "WHERE"),
        AtomRecord("a4", "c3", "d3", "Random astrophysics fact.", "ANY"),
    ]


def test_corpus_failure_when_alias_absent():
    atoms = _atoms()
    gold = GoldRecord("q1", gold_doc_ids=["d_missing"], answer_aliases=["Mars"])
    art = QueryArtifacts("q1", [], [], {})
    row = audit_query(atoms, gold, art)
    assert row.failure_layer == "corpus"
    assert row.gold_atoms_missing_in_corpus is True


def test_atom_extraction_failure_alias_in_corpus_but_not_in_gold_doc():
    atoms = _atoms()
    # Gold doc d3 has no alias-bearing atom; alias "Norway" exists elsewhere.
    gold = GoldRecord("q2", gold_doc_ids=["d3"], answer_aliases=["Norway"])
    art = QueryArtifacts("q2", [], [], {})
    row = audit_query(atoms, gold, art)
    assert row.failure_layer == "atom_extraction"


def test_retrieval_failure_when_gold_atom_not_in_candidates():
    atoms = _atoms()
    gold = GoldRecord("q3", gold_doc_ids=["d1"], answer_aliases=["Norway"])
    art = QueryArtifacts("q3", ["a3", "a4"], [], {})
    row = audit_query(atoms, gold, art)
    assert row.failure_layer == "retrieval"
    assert row.gold_atoms_found == ["a1"]


def test_detection_vs_selection_split():
    atoms = _atoms()
    gold = GoldRecord("q4", gold_doc_ids=["d1"], answer_aliases=["Norway"])
    # In candidates, low score, threshold above score -> detection.
    art_det = QueryArtifacts(
        "q4", ["a1", "a3"], [], {"a1": 0.1}, detector_threshold=0.5
    )
    assert audit_query(atoms, gold, art_det).failure_layer == "detection"
    # In candidates, high score, but not selected -> selection.
    art_sel = QueryArtifacts(
        "q4", ["a1", "a3"], ["a3"], {"a1": 0.9}, detector_threshold=0.5
    )
    assert audit_query(atoms, gold, art_sel).failure_layer == "selection"


def test_generation_failure_when_selected_but_wrong():
    atoms = _atoms()
    gold = GoldRecord("q5", gold_doc_ids=["d1"], answer_aliases=["Norway"])
    art = QueryArtifacts(
        "q5", ["a1"], ["a1"], {"a1": 0.9},
        detector_threshold=0.5, generation_correct=False,
    )
    assert audit_query(atoms, gold, art).failure_layer == "generation"


def test_none_when_correct():
    atoms = _atoms()
    gold = GoldRecord("q6", gold_doc_ids=["d1"], answer_aliases=["Norway"])
    art = QueryArtifacts(
        "q6", ["a1"], ["a1"], {"a1": 0.9},
        detector_threshold=0.5, generation_correct=True,
    )
    assert audit_query(atoms, gold, art).failure_layer == "none"


def test_downgrade_without_aliases():
    atoms = _atoms()
    gold = GoldRecord("q7", gold_doc_ids=["d_unknown"], answer_aliases=[])
    art = QueryArtifacts("q7", [], [], {})
    row = audit_query(atoms, gold, art)
    assert row.failure_layer == "corpus_or_retrieval"
    assert row.notes.get("downgraded") is True


def test_aggregate_summary():
    atoms = _atoms()
    rows = [
        audit_query(
            atoms,
            GoldRecord("q1", ["d1"], ["Norway"]),
            QueryArtifacts("q1", ["a1"], ["a1"], {"a1": 0.9}, 0.5, True),
        ),
        audit_query(
            atoms,
            GoldRecord("q2", ["d1"], ["Norway"]),
            QueryArtifacts("q2", ["a3"], [], {}),
        ),
    ]
    s = aggregate(rows)
    assert s["n_queries"] == 2
    assert s["any_alias_present"] is True
    assert s["failure_layer_distribution"]["none"] == 1
    assert s["failure_layer_distribution"]["retrieval"] == 1
