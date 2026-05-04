import math

from astro_cs_rag.diagnostics.calorimetry import score_shape


def test_compact_shape_classification() -> None:
    scores = [10.0] + [0.1] * 49
    s = score_shape(scores)
    assert s.archetype == "compact"
    assert s.second_peak_gap > 0.95
    assert s.fwhm_index_fraction < 0.05


def test_plateau_shape_classification() -> None:
    # Smooth decay with no isolated peak → plateau / diffuse, not bimodal.
    scores = [1.0 - 0.005 * i for i in range(50)]
    s = score_shape(scores)
    assert s.archetype in {"plateau", "diffuse"}
    assert s.bimodality_coefficient < 5.0 / 9.0


def test_bimodal_shape_classification() -> None:
    scores = [10.0, 9.0] + [0.05] * 30 + [9.0, 9.0] + [0.05] * 16
    s = score_shape(scores)
    assert s.bimodality_coefficient > 0.0
    assert -1e-6 <= s.peak_minus_median  # well defined


def test_empty_returns_empty_archetype() -> None:
    s = score_shape([])
    assert s.archetype == "empty"
    assert s.n_candidates == 0


def test_score_shape_fields_are_finite() -> None:
    s = score_shape([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    for v in (s.peak_score, s.kurtosis, s.skewness, s.bimodality_coefficient,
              s.tail_decay_slope, s.fwhm_index_fraction):
        assert math.isfinite(v)
