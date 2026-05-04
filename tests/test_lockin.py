from astro_cs_rag.retrieval.lockin import (
    coherence_ratio,
    coherent_sum,
    fixed_pattern_phases,
    incoherent_sum,
)


def test_coherent_sum_constructive_with_zero_phases() -> None:
    fields = [
        {"a": 1.0, "b": 0.5},
        {"a": 1.0, "b": 0.5},
    ]
    coh = coherent_sum(fields, phases=[0.0, 0.0])
    inc = incoherent_sum(fields)
    # Coherent: |1+1|^2 = 4. Incoherent: 1^2+1^2 = 2.
    assert coh["a"] == 4.0
    assert inc["a"] == 2.0


def test_coherent_sum_destructive_with_pi_phase() -> None:
    fields = [{"a": 1.0}, {"a": 1.0}]
    coh = coherent_sum(fields, phases=[0.0, 3.141592653589793])
    assert coh["a"] < 1e-9


def test_fixed_pattern_phases_distribute_evenly() -> None:
    phases = fixed_pattern_phases(4)
    assert len(phases) == 4
    assert phases[0] == 0.0


def test_coherence_ratio_handles_zero_intensity() -> None:
    coh = {"a": 0.0}
    inc = {"a": 0.0}
    out = coherence_ratio(coh, inc)
    assert out["a"] == 0.0
