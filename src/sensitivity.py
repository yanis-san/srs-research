"""One-at-a-time sensitivity analysis of the scheduler constants (RQ4).

Each constant is moved by +/-p (20 % by default) while the others stay at
their nominal value, and three things are measured:

- the shift of the scheduled interval on a small set of hand-written
  scenarios;
- the effect on a five-review trajectory (for the ease constants, which do not
  change the current interval);
- for the two latency thresholds, the share of a reference latency set that
  changes category.

This says how much each constant moves the output. It says nothing about
whether the constants are right, and the scenarios are not real data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable


@dataclass(frozen=True)
class AlgorithmParams:
    """Scheduler constants. Defaults match `src/srs_logic.py`."""

    neutral_hrv_rmssd: float = 60.0
    neutral_sleep_score: float = 75.0
    weight_hrv: float = 0.6
    weight_sleep: float = 0.4
    min_bio_multiplier: float = 0.75
    max_bio_multiplier: float = 1.25

    fast_latency_max_ms: int = 2500
    slow_latency_min_ms: int = 6000
    ease_bonus_fast: float = 0.05
    ease_penalty_slow: float = 0.05

    min_ease_factor: float = 1.3
    ease_penalty_on_failure: float = 0.2


# The two functions below re-implement `srs_logic` with injectable constants.
# tests/test_sensitivity.py checks that both agree at the nominal point.


def _bio_multiplier_from_params(
    hrv_rmssd: float | None,
    sleep_score: float | None,
    params: AlgorithmParams,
) -> float:
    total_weight = 0.0
    weighted_delta = 0.0

    if hrv_rmssd is not None and params.neutral_hrv_rmssd != 0:
        hrv_delta = (hrv_rmssd - params.neutral_hrv_rmssd) / params.neutral_hrv_rmssd
        weighted_delta += params.weight_hrv * hrv_delta
        total_weight += params.weight_hrv

    if sleep_score is not None and params.neutral_sleep_score != 0:
        sleep_delta = (sleep_score - params.neutral_sleep_score) / params.neutral_sleep_score
        weighted_delta += params.weight_sleep * sleep_delta
        total_weight += params.weight_sleep

    if total_weight == 0.0:
        return 1.0

    raw = 1.0 + weighted_delta / total_weight
    return max(params.min_bio_multiplier, min(params.max_bio_multiplier, raw))


def interval_from_params(
    *,
    is_correct: bool,
    previous_interval: float,
    current_ease: float,
    response_time_ms: int,
    hrv_rmssd: float | None,
    sleep_score: float | None,
    params: AlgorithmParams,
) -> float:
    """Next interval in days for one review under a given set of constants."""
    if not is_correct:
        return 1.0

    standard = 1.0 if previous_interval == 0 else previous_interval * current_ease
    return max(1.0, standard * _bio_multiplier_from_params(hrv_rmssd, sleep_score, params))


@dataclass(frozen=True)
class Scenario:
    name: str
    is_correct: bool
    previous_interval: float
    current_ease: float
    response_time_ms: int
    hrv_rmssd: float | None
    sleep_score: float | None


# Six hand-written cases. "optimal" and "fatigue" are extreme on purpose and
# hit the multiplier bounds, which is why the bounds come out as influential.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("optimal", True, 10.0, 2.5, 2000, 100.0, 100.0),
    Scenario("neutral", True, 10.0, 2.5, 4000, 60.0, 75.0),
    Scenario("fatigue", True, 10.0, 2.5, 7000, 20.0, 40.0),
    Scenario("new_card_fast", True, 0.0, 2.5, 1500, 80.0, 85.0),
    Scenario("failure_optimal", False, 30.0, 2.5, 3000, 100.0, 100.0),
    Scenario("sensor_missing", True, 15.0, 2.2, 5000, None, None),
)


@dataclass(frozen=True)
class Factor:
    """A constant that can be perturbed."""

    name: str
    nominal: float
    apply: Callable[[AlgorithmParams, float], AlgorithmParams]


def _build_factors() -> list[Factor]:
    base = AlgorithmParams()

    def make_factor(field_name: str) -> Factor:
        nominal_value = getattr(base, field_name)

        def apply(params: AlgorithmParams, value: float) -> AlgorithmParams:
            # Latency thresholds are integers.
            new_value = int(round(value)) if isinstance(nominal_value, int) else value
            return replace(params, **{field_name: new_value})

        return Factor(name=field_name, nominal=float(nominal_value), apply=apply)

    return [
        make_factor(name)
        for name in (
            "neutral_hrv_rmssd",
            "neutral_sleep_score",
            "weight_hrv",
            "weight_sleep",
            "min_bio_multiplier",
            "max_bio_multiplier",
            "fast_latency_max_ms",
            "slow_latency_min_ms",
            "ease_bonus_fast",
            "ease_penalty_slow",
            "min_ease_factor",
            "ease_penalty_on_failure",
        )
    ]


FACTORS: tuple[Factor, ...] = tuple(_build_factors())


@dataclass(frozen=True)
class FactorSensitivity:
    """Result for one constant.

    mean_interval_delta_plus/minus: mean signed interval change (days) at +p / -p.
    max_abs_delta: largest absolute change over all scenarios (days).
    mean_elasticity: mean |relative change of interval| / p.
    timeline_impact: largest gap on the five-review trajectory, relative to
        the last nominal interval.
    reclassification_rate: latency thresholds only; 0.0 for other constants.
    """

    name: str
    nominal: float
    mean_interval_delta_plus: float
    mean_interval_delta_minus: float
    max_abs_delta: float
    mean_elasticity: float
    timeline_impact: float
    reclassification_rate: float
    is_influential: bool
    note: str


@dataclass(frozen=True)
class SensitivityReport:
    perturbation: float
    # Mean Spearman rho between nominal and perturbed intervals.
    order_preserved_ratio: float
    factors: tuple[FactorSensitivity, ...]
    summary: str


def _perturbed_params(factor: Factor, perturbation: float) -> AlgorithmParams:
    # Floored so a weight or threshold never becomes zero or negative.
    value = max(factor.nominal * (1.0 + perturbation), 1e-6)
    return factor.apply(AlgorithmParams(), value)


def _scenario_interval(scenario: Scenario, params: AlgorithmParams) -> float:
    return interval_from_params(
        is_correct=scenario.is_correct,
        previous_interval=scenario.previous_interval,
        current_ease=scenario.current_ease,
        response_time_ms=scenario.response_time_ms,
        hrv_rmssd=scenario.hrv_rmssd,
        sleep_score=scenario.sleep_score,
        params=params,
    )


def interval_delta(scenario: Scenario, factor: Factor, perturbation: float) -> float:
    """Perturbed minus nominal interval, in days."""
    nominal = _scenario_interval(scenario, AlgorithmParams())
    perturbed = _scenario_interval(scenario, _perturbed_params(factor, perturbation))
    return perturbed - nominal


def _elasticity(scenario: Scenario, factor: Factor, perturbation: float) -> float:
    nominal = _scenario_interval(scenario, AlgorithmParams())
    if nominal == 0.0:
        return 0.0
    return (interval_delta(scenario, factor, perturbation) / nominal) / perturbation


def analyze_factor(
    factor: Factor,
    scenarios: tuple[Scenario, ...],
    perturbation: float,
    influence_threshold: float,
    timeline_threshold: float = 0.05,
    reclassification_threshold: float = 0.05,
) -> FactorSensitivity:
    """A factor is flagged influential if any of the three measures passes its threshold."""
    deltas_plus = [interval_delta(s, factor, +perturbation) for s in scenarios]
    deltas_minus = [interval_delta(s, factor, -perturbation) for s in scenarios]
    elasticities = [round(abs(_elasticity(s, factor, perturbation)), 4) for s in scenarios]
    max_abs = max((abs(d) for d in deltas_plus + deltas_minus), default=0.0)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    timeline = round(timeline_sensitivity(factor, perturbation), 4)
    reclassification = 0.0
    if factor.name in THRESHOLD_FACTORS:
        reclassification = round(threshold_reclassification_rate(factor.name, perturbation), 4)

    if max_abs >= influence_threshold:
        note = "shifts the interval directly"
    elif reclassification >= reclassification_threshold:
        note = f"{reclassification * 100:.0f} % of reference latencies change category"
    elif timeline >= timeline_threshold:
        note = "affects later reviews through ease"
    else:
        note = "small effect"

    return FactorSensitivity(
        name=factor.name,
        nominal=factor.nominal,
        mean_interval_delta_plus=round(mean(deltas_plus), 4),
        mean_interval_delta_minus=round(mean(deltas_minus), 4),
        max_abs_delta=round(max_abs, 4),
        mean_elasticity=round(mean(elasticities), 4),
        timeline_impact=timeline,
        reclassification_rate=reclassification,
        is_influential=(
            max_abs >= influence_threshold
            or timeline >= timeline_threshold
            or reclassification >= reclassification_threshold
        ),
        note=note,
    )


def _spearman_rank_correlation(a: list[float], b: list[float]) -> float:
    """Spearman rho with average ranks for ties. Returns 1.0 for n < 2 or constant input."""
    n = len(a)
    if n != len(b) or n < 2:
        return 1.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                result[order[k]] = average_rank
            i = j + 1
        return result

    rank_a = ranks(a)
    rank_b = ranks(b)
    mean_a = sum(rank_a) / n
    mean_b = sum(rank_b) / n
    cov = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(rank_a, rank_b))
    var_a = sum((ra - mean_a) ** 2 for ra in rank_a)
    var_b = sum((rb - mean_b) ** 2 for rb in rank_b)
    if var_a == 0.0 or var_b == 0.0:
        return 1.0
    return cov / (var_a * var_b) ** 0.5


def simulate_ease_trajectory(
    base_ease: float,
    sequence: list[str],
    *,
    previous_interval: float = 10.0,
    params: AlgorithmParams | None = None,
) -> list[float]:
    """Intervals over a run of successful reviews with neutral HRV/sleep.

    `sequence` gives the latency category of each review ("fast", "normal",
    "slow"). Used for the ease constants, which only act on later reviews.
    """
    if params is None:
        params = AlgorithmParams()

    ease = base_ease
    interval = previous_interval
    intervals: list[float] = []

    for latency in sequence:
        new_interval = max(1.0, interval * ease if interval > 0 else 1.0)
        intervals.append(new_interval)

        if latency == "fast":
            ease += params.ease_bonus_fast
        elif latency == "slow":
            ease = max(params.min_ease_factor, ease - params.ease_penalty_slow)

        interval = new_interval

    return intervals


def timeline_sensitivity(
    factor: Factor,
    perturbation: float,
    sequence: list[str] | None = None,
    base_ease: float = 2.5,
) -> float:
    """Largest gap between nominal and perturbed trajectories, relative to the last nominal interval."""
    if sequence is None:
        sequence = ["fast", "normal", "slow", "fast", "normal"]

    nominal = simulate_ease_trajectory(base_ease, sequence, params=AlgorithmParams())
    perturbed = simulate_ease_trajectory(
        base_ease, sequence, params=_perturbed_params(factor, perturbation)
    )

    max_gap = max(abs(p - n) for n, p in zip(nominal, perturbed))
    last_nominal = nominal[-1] if nominal[-1] != 0 else 1.0
    return max_gap / last_nominal


def rank_robustness(
    scenarios: tuple[Scenario, ...],
    factor: Factor,
    perturbation: float,
) -> float:
    """Spearman rho between nominal and perturbed intervals across scenarios.

    With six scenarios, two of which tie at 1 day, this is a coarse check.
    """
    nominal_params = AlgorithmParams()
    perturbed_params = _perturbed_params(factor, perturbation)
    nominal = [_scenario_interval(s, nominal_params) for s in scenarios]
    perturbed = [_scenario_interval(s, perturbed_params) for s in scenarios]
    return _spearman_rank_correlation(nominal, perturbed)


# Thirty latencies (ms) picked by hand to spread over the three categories.
# They are not measured data; the reclassification rates depend on this choice
# (e.g. 2/30 = 6.7 %, 4/30 = 13.3 %).
REFERENCE_LATENCIES_MS: tuple[int, ...] = (
    500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300,
    2600, 2800, 3100, 3400, 3700, 4000, 4300, 4600, 4900, 5200, 5500, 5800,
    6200, 6600, 7100, 7700, 8400, 9200, 10000, 11000,
)


def classify_latency_with_thresholds(
    response_time_ms: int,
    fast_threshold: int,
    slow_threshold: int,
) -> str:
    if response_time_ms < fast_threshold:
        return "fast"
    if response_time_ms > slow_threshold:
        return "slow"
    return "normal"


def threshold_reclassification_rate(
    threshold_name: str,
    disturbance: float,
    latencies: tuple[int, ...] = REFERENCE_LATENCIES_MS,
) -> float:
    """Share of `latencies` that change category, worst of +disturbance and -disturbance."""
    base = AlgorithmParams()
    nominal_fast = base.fast_latency_max_ms
    nominal_slow = base.slow_latency_min_ms

    def rate_for(direction: float) -> float:
        fast = nominal_fast
        slow = nominal_slow
        if threshold_name == "fast_latency_max_ms":
            fast = max(1, int(round(nominal_fast * (1.0 + direction))))
        elif threshold_name == "slow_latency_min_ms":
            slow = max(nominal_fast + 1, int(round(nominal_slow * (1.0 + direction))))
        else:
            return 0.0

        changed = sum(
            classify_latency_with_thresholds(ms, nominal_fast, nominal_slow)
            != classify_latency_with_thresholds(ms, fast, slow)
            for ms in latencies
        )
        return changed / len(latencies)

    return max(rate_for(+disturbance), rate_for(-disturbance))


THRESHOLD_FACTORS: frozenset[str] = frozenset({"fast_latency_max_ms", "slow_latency_min_ms"})


def run_sensitivity_analysis(
    perturbation: float = 0.2,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    influence_threshold: float = 1.0,
    timeline_threshold: float = 0.05,
    reclassification_threshold: float = 0.05,
) -> SensitivityReport:
    """Analyse every factor; results sorted by largest interval shift."""
    results: list[FactorSensitivity] = []
    rhos: list[float] = []
    for factor in FACTORS:
        result = analyze_factor(
            factor,
            scenarios,
            perturbation,
            influence_threshold,
            timeline_threshold,
            reclassification_threshold,
        )
        rho = rank_robustness(scenarios, factor, perturbation)
        rhos.append(rho)
        results.append(replace(result, note=f"{result.note} (rho={rho:.2f})"))

    results.sort(key=lambda r: (r.max_abs_delta, r.timeline_impact), reverse=True)
    mean_rho = sum(rhos) / len(rhos) if rhos else 1.0

    influential = [r.name for r in results if r.is_influential]
    if influential:
        summary = (
            f"At +/-{perturbation * 100:.0f} %, influential constants: "
            f"{', '.join(influential)}. Mean Spearman rho = {mean_rho:.3f}."
        )
    else:
        summary = (
            f"At +/-{perturbation * 100:.0f} %, no constant passes the thresholds. "
            f"Mean Spearman rho = {mean_rho:.3f}."
        )

    return SensitivityReport(
        perturbation=perturbation,
        order_preserved_ratio=round(mean_rho, 4),
        factors=tuple(results),
        summary=summary,
    )
