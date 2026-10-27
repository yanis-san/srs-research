"""Tests of the sensitivity analysis (src/sensitivity.py)."""

from __future__ import annotations

import pytest

from src.sensitivity import (
    DEFAULT_SCENARIOS,
    FACTORS,
    REFERENCE_LATENCIES_MS,
    AlgorithmParams,
    Scenario,
    _spearman_rank_correlation,
    classify_latency_with_thresholds,
    interval_from_params,
    rank_robustness,
    run_sensitivity_analysis,
    threshold_reclassification_rate,
    timeline_sensitivity,
)
from src.srs_logic import calculate_biometric_srs


# ---------------------------------------------------------------------------
# Equivalence with the production algorithm
# ---------------------------------------------------------------------------


def test_interval_from_params_equivalent_at_nominal_point() -> None:
    """The re-implementation must match `calculate_biometric_srs` at nominal values."""
    params = AlgorithmParams()

    # Representative case: mature card, success, favourable biometrics.
    parametrized = interval_from_params(
        is_correct=True,
        previous_interval=10.0,
        current_ease=2.5,
        response_time_ms=4000,
        hrv_rmssd=85.0,
        sleep_score=90.0,
        params=params,
    )
    production = calculate_biometric_srs(
        is_correct=True,
        previous_interval=10.0,
        current_ease=2.5,
        response_time_ms=4000,
        hrv_rmssd=85.0,
        sleep_score=90.0,
    ).new_interval

    # Rounding tolerance: production rounds to 2 decimals.
    assert parametrized == pytest.approx(production, abs=1e-2)


def test_interval_from_params_failure_returns_one_day() -> None:
    """A failure always schedules a 1-day review, whatever the biometrics."""
    params = AlgorithmParams()
    value = interval_from_params(
        is_correct=False,
        previous_interval=30.0,
        current_ease=2.5,
        response_time_ms=3000,
        hrv_rmssd=100.0,
        sleep_score=100.0,
        params=params,
    )
    assert value == 1.0


def test_multiplier_clamped_by_max() -> None:
    """An extreme biometrics saturates at the `max_bio_multiplier` ceiling.

    We force a huge HRV and sleep: the multiplier must be capped at
    max_bio_multiplier (1.25), translated into interval base × 1.25
    (base = 10 × 2.5 = 25 -> 31.25).
    """
    params = AlgorithmParams(max_bio_multiplier=1.25)
    value = interval_from_params(
        is_correct=True,
        previous_interval=10.0,
        current_ease=2.5,
        response_time_ms=4000,
        hrv_rmssd=500.0,
        sleep_score=500.0,
        params=params,
    )
    # base = 25 ; × 1.25 = 31.25.
    assert value == pytest.approx(31.25, abs=1e-6)


def test_multiplier_clamped_by_min() -> None:
    """A catastrophic biometrics saturates at the `min_bio_multiplier` floor."""
    params = AlgorithmParams(min_bio_multiplier=0.75)
    value = interval_from_params(
        is_correct=True,
        previous_interval=10.0,
        current_ease=2.5,
        response_time_ms=4000,
        hrv_rmssd=0.0,
        sleep_score=0.0,
        params=params,
    )
    # base = 25 ; × 0.75 = 18.75.
    assert value == pytest.approx(18.75, abs=1e-6)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------


def test_spearman_identical_rankings_equals_one() -> None:
    """Two strictly increasing lists of the same order -> rho = 1.0."""
    assert _spearman_rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == (
        pytest.approx(1.0)
    )


def test_spearman_inverse_order_equals_minus_one() -> None:
    """An inverted order -> rho = -1.0."""
    assert _spearman_rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == (
        pytest.approx(-1.0)
    )


def test_spearman_ties_handled() -> None:
    """Ties must not raise an error (average rank)."""
    # Two identical values in the middle: the correlation stays high.
    rho = _spearman_rank_correlation([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 2.0, 4.0])
    assert rho == pytest.approx(1.0)


def test_spearman_too_short_list_returns_one() -> None:
    """Fewer than two points: by convention, perfectly correlated."""
    assert _spearman_rank_correlation([1.0], [2.0]) == 1.0


# ---------------------------------------------------------------------------
# Latency threshold reclassification
# ---------------------------------------------------------------------------


def test_classification_with_injectable_thresholds() -> None:
    """Check the classification boundaries (before/after thresholds)."""
    # fast_threshold=2500, slow_threshold=6000.
    assert classify_latency_with_thresholds(2000, 2500, 6000) == "fast"
    assert classify_latency_with_thresholds(4000, 2500, 6000) == "normal"
    assert classify_latency_with_thresholds(7000, 2500, 6000) == "slow"


def test_zero_reclassification_when_threshold_unperturbed() -> None:
    """With a null perturbation, no item changes category."""
    rate = threshold_reclassification_rate("fast_latency_max_ms", disturbance=0.0)
    assert rate == pytest.approx(0.0)


def test_reclassification_grows_with_perturbation() -> None:
    """Moving a threshold further reclassifies at least as many items.

    Monotonicity property: a larger perturbation cannot reclassify FEWER items
    than a smaller perturbation.
    """
    small = threshold_reclassification_rate("slow_latency_min_ms", 0.1)
    large = threshold_reclassification_rate("slow_latency_min_ms", 0.4)
    assert large >= small


def test_reclassification_within_bounds() -> None:
    """The reclassification rate is always a proportion in [0, 1]."""
    for name in ("fast_latency_max_ms", "slow_latency_min_ms"):
        rate = threshold_reclassification_rate(name, disturbance=0.2)
        assert 0.0 <= rate <= 1.0


def test_reference_latency_distribution_not_empty() -> None:
    """The reference distribution must contain values (otherwise division by 0)."""
    assert len(REFERENCE_LATENCIES_MS) > 0
    assert all(ms > 0 for ms in REFERENCE_LATENCIES_MS)


# ---------------------------------------------------------------------------
# Trajectory and robustness
# ---------------------------------------------------------------------------


def test_timeline_sensitivity_non_negative() -> None:
    """The trajectory impact is a normalised absolute value (>= 0)."""
    bonus_factor = next(f for f in FACTORS if f.name == "ease_bonus_fast")
    impact = timeline_sensitivity(bonus_factor, perturbation=0.2)
    assert impact >= 0.0


def test_rank_robustness_perfect_on_ineffective_factor() -> None:
    """A factor with no direct effect must perfectly preserve the ranking.

    Example: `min_ease_factor`, with no effect on the current interval, must
    not reorder the scenarios -> rho = 1.0.
    """
    factor = next(f for f in FACTORS if f.name == "min_ease_factor")
    rho = rank_robustness(DEFAULT_SCENARIOS, factor, perturbation=0.2)
    assert rho == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def test_run_sensitivity_analysis_structure() -> None:
    """The report must contain one result per factor, sorted, plus a summary."""
    report = run_sensitivity_analysis(perturbation=0.2)

    # One result per factor.
    assert len(report.factors) == len(FACTORS)

    # Descending sort by maximum effect on the interval.
    deltas = [f.max_abs_delta for f in report.factors]
    assert deltas == sorted(deltas, reverse=True)

    # The global robustness is a valid proportion.
    assert 0.0 <= report.order_preserved_ratio <= 1.0

    # The summary is non-empty.
    assert isinstance(report.summary, str)
    assert report.summary


def test_run_sensitivity_analysis_flags_critical_bounds() -> None:
    """With the default scenarios, both multiplier bounds are flagged."""
    report = run_sensitivity_analysis(perturbation=0.2)
    influential_names = {f.name for f in report.factors if f.is_influential}
    assert "max_bio_multiplier" in influential_names
    assert "min_bio_multiplier" in influential_names


def test_run_sensitivity_analysis_reclassified_thresholds_are_influential() -> None:
    """Thresholds do not shift intervals directly; reclassification must catch them."""
    report = run_sensitivity_analysis(perturbation=0.2)
    fast = next(f for f in report.factors if f.name == "fast_latency_max_ms")
    slow = next(f for f in report.factors if f.name == "slow_latency_min_ms")
    # They must have a reclassification rate > 0 and be influential.
    assert fast.reclassification_rate > 0.0
    assert slow.reclassification_rate > 0.0
    assert fast.is_influential
    assert slow.is_influential


def test_run_sensitivity_analysis_custom_perturbation() -> None:
    """The report reflects the requested perturbation amplitude."""
    report = run_sensitivity_analysis(perturbation=0.3)
    assert report.perturbation == pytest.approx(0.3)
