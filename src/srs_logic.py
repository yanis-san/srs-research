"""Scheduling logic: pure functions, no database, no clock.

All constants below are working assumptions, not values taken from the
literature. Calibrating them is one of the goals of the planned study.
"""

from __future__ import annotations

from dataclasses import dataclass

# SM-2 style ease factor.
DEFAULT_EASE_FACTOR: float = 2.5
MIN_EASE_FACTOR: float = 1.3
EASE_PENALTY_ON_FAILURE: float = 0.2

# Latency thresholds (ms) and the ease adjustment they trigger on a success.
FAST_LATENCY_MAX_MS: int = 2500
SLOW_LATENCY_MIN_MS: int = 6000
EASE_BONUS_FAST: float = 0.05
EASE_PENALTY_SLOW: float = 0.05

# Reference values at which the multiplier is exactly 1.0.
NEUTRAL_HRV_RMSSD: float = 60.0
NEUTRAL_SLEEP_SCORE: float = 75.0

# Bounds and weights of the multiplier.
MIN_BIO_MULTIPLIER: float = 0.75
MAX_BIO_MULTIPLIER: float = 1.25
WEIGHT_HRV: float = 0.6
WEIGHT_SLEEP: float = 0.4

LATENCY_CATEGORY_FAST: str = "Fast"
LATENCY_CATEGORY_NORMAL: str = "Normal"
LATENCY_CATEGORY_SLOW: str = "Slow"


def classify_latency(response_time_ms: int) -> str:
    """Fast below 2500 ms, slow above 6000 ms, normal otherwise.

    Both bounds belong to "Normal": 2500 ms and 6000 ms are normal.
    """
    if response_time_ms < FAST_LATENCY_MAX_MS:
        return LATENCY_CATEGORY_FAST
    if response_time_ms > SLOW_LATENCY_MIN_MS:
        return LATENCY_CATEGORY_SLOW
    return LATENCY_CATEGORY_NORMAL


def calculate_bio_multiplier(
    hrv_rmssd: float | None,
    sleep_score: float | None,
) -> float:
    """Weighted relative deviation from the reference values, clamped.

        m = 1 + [w_hrv * (hrv - 60) / 60 + w_sleep * (sleep - 75) / 75] / (sum of weights used)

    A missing signal is left out and the remaining weight is renormalised, so
    a single signal still gives a multiplier centred on 1.0. With no signal at
    all the multiplier is 1.0.
    """
    total_weight = 0.0
    weighted_delta = 0.0

    if hrv_rmssd is not None:
        weighted_delta += WEIGHT_HRV * (hrv_rmssd - NEUTRAL_HRV_RMSSD) / NEUTRAL_HRV_RMSSD
        total_weight += WEIGHT_HRV

    if sleep_score is not None:
        weighted_delta += (
            WEIGHT_SLEEP * (sleep_score - NEUTRAL_SLEEP_SCORE) / NEUTRAL_SLEEP_SCORE
        )
        total_weight += WEIGHT_SLEEP

    if total_weight == 0.0:
        return 1.0

    raw = 1.0 + weighted_delta / total_weight
    return max(MIN_BIO_MULTIPLIER, min(MAX_BIO_MULTIPLIER, raw))


def compute_standard_anki_interval(
    previous_interval: float,
    current_ease: float,
    is_correct: bool,
) -> float:
    """Interval the base scheduler gives without latency or physiology.

    Failure or new card: 1 day. Otherwise: previous interval x ease.
    """
    if not is_correct or previous_interval == 0:
        return 1.0
    return previous_interval * current_ease


@dataclass(frozen=True)
class BiometricSrsResult:
    new_interval: float
    new_ease: float
    bio_multiplier: float
    standard_anki_interval: float
    latency_category: str
    explanation: str


def format_explanation(
    is_correct: bool,
    previous_interval: float,
    new_interval: float,
    standard_anki_interval: float,
    bio_multiplier: float,
    latency_category: str,
    response_time_ms: int,
    hrv_rmssd: float | None,
    sleep_score: float | None,
) -> str:
    """One-line summary of how the interval was obtained."""
    result_label = "Success" if is_correct else "Failure"

    signals = []
    if hrv_rmssd is not None:
        signals.append(f"HRV {hrv_rmssd:.0f} ms")
    if sleep_score is not None:
        signals.append(f"sleep {sleep_score:.0f}/100")
    signal_text = ", ".join(signals) if signals else "no HRV or sleep data"

    pct = (bio_multiplier - 1.0) * 100.0
    if pct > 0:
        multiplier_text = f"+{pct:.0f} % (above reference values)"
    elif pct < 0:
        multiplier_text = f"{pct:.0f} % (below reference values)"
    else:
        multiplier_text = "no change"

    return (
        f"{result_label} in {response_time_ms} ms ({latency_category}). "
        f"{signal_text}: multiplier x{bio_multiplier:.2f}, {multiplier_text}. "
        f"Interval {previous_interval:.1f} d -> {new_interval:.1f} d "
        f"(base scheduler: {standard_anki_interval:.1f} d)."
    )


def calculate_biometric_srs(
    is_correct: bool,
    previous_interval: float,
    current_ease: float,
    response_time_ms: int,
    hrv_rmssd: float | None,
    sleep_score: float | None,
) -> BiometricSrsResult:
    """Compute the next interval and ease for one review.

    1. Base interval (see `compute_standard_anki_interval`).
    2. On a success, the interval is multiplied by the HRV/sleep multiplier
       (minimum 1 day). On a failure it stays at 1 day and ease drops by 0.2.
    3. On a success only, latency adjusts ease: +0.05 if fast, -0.05 if slow.
       A fast wrong answer is not rewarded.
    """
    standard_interval = compute_standard_anki_interval(
        previous_interval=previous_interval,
        current_ease=current_ease,
        is_correct=is_correct,
    )
    bio_multiplier = calculate_bio_multiplier(hrv_rmssd, sleep_score)
    latency_category = classify_latency(response_time_ms)

    if is_correct:
        new_interval = max(1.0, standard_interval * bio_multiplier)
        new_ease = current_ease
        if latency_category == LATENCY_CATEGORY_FAST:
            new_ease += EASE_BONUS_FAST
        elif latency_category == LATENCY_CATEGORY_SLOW:
            new_ease = max(MIN_EASE_FACTOR, new_ease - EASE_PENALTY_SLOW)
    else:
        new_interval = 1.0
        new_ease = max(MIN_EASE_FACTOR, current_ease - EASE_PENALTY_ON_FAILURE)

    # Rounded to avoid values like 2.3499999999999996 in the UI.
    rounded_interval = round(new_interval, 2)
    rounded_multiplier = round(bio_multiplier, 4)
    rounded_standard = round(standard_interval, 2)

    explanation = format_explanation(
        is_correct=is_correct,
        previous_interval=previous_interval,
        new_interval=rounded_interval,
        standard_anki_interval=rounded_standard,
        bio_multiplier=rounded_multiplier,
        latency_category=latency_category,
        response_time_ms=response_time_ms,
        hrv_rmssd=hrv_rmssd,
        sleep_score=sleep_score,
    )

    return BiometricSrsResult(
        new_interval=rounded_interval,
        new_ease=round(new_ease, 4),
        bio_multiplier=rounded_multiplier,
        standard_anki_interval=rounded_standard,
        latency_category=latency_category,
        explanation=explanation,
    )
