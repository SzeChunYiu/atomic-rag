from astro_cs_rag.evaluation.cross_section import (
    compute_sigma_efficiency,
    compute_sigma_recall,
)


def test_sigma_recall_factors_out_query_luminosity() -> None:
    per_q_recall = {"q1": 1.0, "q2": 0.5, "q3": 0.0}
    per_q_n_gold = {"q1": 1, "q2": 2, "q3": 3}
    sigma, lumi, n = compute_sigma_recall(
        per_query_recall=per_q_recall,
        per_query_n_gold=per_q_n_gold,
        k=5,
    )
    # Expected luminosity = min(1,5)+min(2,5)+min(3,5) = 6
    # Weighted recall = 1*1 + 2*0.5 + 3*0 = 2 → sigma = 2/6 ≈ 0.333
    assert lumi == 6.0
    assert n == 3
    assert abs(sigma - (2.0 / 6.0)) < 1e-9


def test_sigma_efficiency_handles_zero_recall() -> None:
    assert compute_sigma_efficiency(mean_answer_metric=0.5, mean_recall_at_k=0.0) is None


def test_sigma_efficiency_normal() -> None:
    eff = compute_sigma_efficiency(mean_answer_metric=0.40, mean_recall_at_k=0.80)
    assert abs(eff - 0.5) < 1e-9
