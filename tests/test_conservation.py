from astro_cs_rag.evaluation.conservation import (
    conservation_residuals,
    entity_residual,
    numeric_residual,
    temporal_residual,
)


def test_entity_residual_zero_when_supported() -> None:
    answer = "The Crab Nebula contains a pulsar."
    evidence = "The Crab Nebula is a supernova remnant containing a pulsar."
    assert entity_residual(answer, evidence) == 0.0


def test_entity_residual_high_when_unsupported() -> None:
    answer = "NASA observed M87 with the Hubble telescope."
    evidence = "Cosmic rays are charged particles."
    r = entity_residual(answer, evidence)
    assert r > 0.5


def test_numeric_residual_close_match() -> None:
    answer = "The CMB temperature is 2.7 K."
    evidence = "Cosmic microwave background radiation is approximately 2.725 K."
    r = numeric_residual(answer, evidence, tol_rel=0.05)
    assert r == 0.0


def test_numeric_residual_disagreement() -> None:
    answer = "The CMB temperature is 5000 K."
    evidence = "Cosmic microwave background radiation is approximately 2.725 K."
    r = numeric_residual(answer, evidence, tol_rel=0.05)
    assert r > 0.0


def test_temporal_residual_consistent_order() -> None:
    answer = "First 1054 then 1968 events occurred."
    evidence = "Events at 1054 preceded those at 1968."
    assert temporal_residual(answer, evidence) == 0.0


def test_full_residuals_sum_to_faithfulness() -> None:
    cr = conservation_residuals("Random answer.", ["Unrelated evidence."])
    assert 0.0 <= cr.mean <= 1.0
    assert 0.0 <= cr.faithfulness <= 1.0
    assert abs(cr.faithfulness + cr.mean - 1.0) < 1e-9
