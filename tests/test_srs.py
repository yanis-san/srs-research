"""Tests of the scheduling logic (no I/O) and of the API (in-memory SQLite)."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src import main
from src.database import Base, Flashcard
from src.srs_logic import (
    LATENCY_CATEGORY_FAST,
    LATENCY_CATEGORY_NORMAL,
    LATENCY_CATEGORY_SLOW,
    MAX_BIO_MULTIPLIER,
    MIN_BIO_MULTIPLIER,
    calculate_bio_multiplier,
    calculate_biometric_srs,
    classify_latency,
    compute_standard_anki_interval,
)


# ===========================================================================
# Part 1: pure logic tests of the SRS (no I/O)
# ===========================================================================
class TestClassifyLatency:
    """Check the exact bounds of the cognitive latency categories."""

    def test_very_fast_response_is_automatic(self) -> None:
        # 100 ms: clearly below 2500 ms.
        assert classify_latency(100) == LATENCY_CATEGORY_FAST

    def test_just_below_fast_bound(self) -> None:
        # 2499 ms: still "Fast".
        assert classify_latency(2499) == LATENCY_CATEGORY_FAST

    def test_exact_fast_bound_is_normal(self) -> None:
        # 2500 ms is the bound: it falls in "Normal" (>= bound).
        assert classify_latency(2500) == LATENCY_CATEGORY_NORMAL

    def test_normal_zone(self) -> None:
        assert classify_latency(4000) == LATENCY_CATEGORY_NORMAL

    def test_exact_slow_bound_is_normal(self) -> None:
        # 6000 ms stays "Normal": "Slow" starts strictly beyond.
        assert classify_latency(6000) == LATENCY_CATEGORY_NORMAL

    def test_beyond_slow_bound(self) -> None:
        assert classify_latency(6001) == LATENCY_CATEGORY_SLOW


class TestBioMultiplier:
    """Check the Cognitive Availability Multiplier and its clamping."""

    def test_neutral_references_give_one(self) -> None:
        # HRV = 60 and sleep = 75 are the references -> exactly 1.0.
        assert calculate_bio_multiplier(60.0, 75.0) == pytest.approx(1.0)

    def test_no_data_gives_one(self) -> None:
        # Without sensors, the state is strictly neutral.
        assert calculate_bio_multiplier(None, None) == pytest.approx(1.0)

    def test_optimal_state_reaches_ceiling(self) -> None:
        # Very high HRV + perfect sleep -> clamped to 1.25.
        assert calculate_bio_multiplier(100.0, 100.0) == pytest.approx(
            MAX_BIO_MULTIPLIER
        )

    def test_exhausted_state_reaches_floor(self) -> None:
        # Low HRV + bad sleep -> clamped to 0.75.
        assert calculate_bio_multiplier(20.0, 40.0) == pytest.approx(
            MIN_BIO_MULTIPLIER
        )

    def test_good_state_increases_multiplier(self) -> None:
        # Above the references, the multiplier must exceed 1.0.
        assert calculate_bio_multiplier(80.0, 85.0) > 1.0

    def test_bad_state_reduces_multiplier(self) -> None:
        # Below the references, it must be lower than 1.0.
        assert calculate_bio_multiplier(45.0, 60.0) < 1.0

    def test_single_field_is_renormalised(self) -> None:
        # HRV only -> relative gap (90-60)/60 = 0.5 -> 1 + 0.5 = 1.5 clamped 1.25.
        assert calculate_bio_multiplier(90.0, None) == pytest.approx(
            MAX_BIO_MULTIPLIER
        )

    def test_multiplier_always_within_bounds(self) -> None:
        # Robustness test: sweep of extreme values.
        for hrv in (0.0, 30.0, 60.0, 120.0, 500.0):
            for sleep in (0.0, 50.0, 75.0, 100.0):
                value = calculate_bio_multiplier(hrv, sleep)
                assert MIN_BIO_MULTIPLIER <= value <= MAX_BIO_MULTIPLIER


class TestStandardAnkiInterval:
    """Check the "classic Anki" baseline used as a comparison."""

    def test_failure_gives_one_day(self) -> None:
        assert compute_standard_anki_interval(10.0, 2.5, is_correct=False) == 1.0

    def test_new_card_gives_one_day(self) -> None:
        assert compute_standard_anki_interval(0.0, 2.5, is_correct=True) == 1.0

    def test_success_multiplies_by_ease(self) -> None:
        # 10 days * 2.5 = 25 days.
        assert compute_standard_anki_interval(10.0, 2.5, is_correct=True) == 25.0


class TestCalculateBiometricSrs:
    """Tests of the algorithm core: complete and consistent result."""

    def test_new_success_in_neutral_state(self) -> None:
        # New card (interval 0), normal answer, neutral biometrics:
        # 1-day interval, unchanged ease (2.5).
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=0.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_interval == pytest.approx(1.0)
        assert result.new_ease == pytest.approx(2.5)
        assert result.bio_multiplier == pytest.approx(1.0)
        assert result.latency_category == LATENCY_CATEGORY_NORMAL

    def test_failure_resets_and_penalises(self) -> None:
        # Failure: interval reset to 1 day whatever the biometrics, ease
        # 2.5 -> 2.3.
        result = calculate_biometric_srs(
            is_correct=False,
            previous_interval=30.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=100.0,
            sleep_score=100.0,
        )
        assert result.new_interval == pytest.approx(1.0)
        assert result.new_ease == pytest.approx(2.3)

    def test_failure_does_not_go_below_floor(self) -> None:
        # Ease at 1.3 minus 0.2 must never go below 1.3.
        result = calculate_biometric_srs(
            is_correct=False,
            previous_interval=5.0,
            current_ease=1.3,
            response_time_ms=4000,
            hrv_rmssd=None,
            sleep_score=None,
        )
        assert result.new_ease == pytest.approx(1.3)

    def test_fast_latency_ease_bonus(self) -> None:
        # Response < 2500 ms on a success -> ease +0.05 (2.5 -> 2.55).
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=2000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_ease == pytest.approx(2.55)
        assert result.latency_category == LATENCY_CATEGORY_FAST

    def test_slow_latency_ease_penalty(self) -> None:
        # Response > 6000 ms on a success -> ease -0.05 (2.5 -> 2.45).
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=7000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_ease == pytest.approx(2.45)
        assert result.latency_category == LATENCY_CATEGORY_SLOW

    def test_fast_failure_gets_no_ease_bonus(self) -> None:
        # A quick wrong answer is still a failure: only the -0.2 penalty applies.
        result = calculate_biometric_srs(
            is_correct=False,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=1000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_ease == pytest.approx(2.3)

    def test_slow_failure_gets_no_extra_penalty(self) -> None:
        result = calculate_biometric_srs(
            is_correct=False,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=9000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_ease == pytest.approx(2.3)

    def test_normal_latency_no_effect_on_ease(self) -> None:
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_ease == pytest.approx(2.5)

    def test_optimal_biometrics_extend_interval(self) -> None:
        # 10 d * 2.5 (standard) = 25 d, * 1.25 (multiplier) = 31.25 d.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=100.0,
            sleep_score=100.0,
        )
        assert result.standard_anki_interval == pytest.approx(25.0)
        assert result.new_interval == pytest.approx(31.25)
        assert result.new_interval > result.standard_anki_interval

    def test_fatigue_biometrics_shorten_interval(self) -> None:
        # 25 d * 0.75 = 18.75 d: the biometric interval is shorter.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=20.0,
            sleep_score=40.0,
        )
        assert result.new_interval == pytest.approx(18.75)
        assert result.new_interval < result.standard_anki_interval

    def test_neutral_state_equals_classic_anki(self) -> None:
        # Without biometric or latency effect, the result must equal Anki.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert result.new_interval == pytest.approx(result.standard_anki_interval)

    def test_interval_never_below_one_day_if_success(self) -> None:
        # Even with a minimal multiplier and a standard interval of 1, the
        # interval must not fall below 1 day.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=0.0,
            current_ease=2.5,
            response_time_ms=4000,
            hrv_rmssd=20.0,
            sleep_score=40.0,
        )
        assert result.new_interval >= 1.0

    def test_explanation_is_a_non_empty_sentence(self) -> None:
        # The explanatory sentence must contain the key information.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=2000,
            hrv_rmssd=90.0,
            sleep_score=85.0,
        )
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
        assert "Success" in result.explanation
        # The measured latency must appear in the explanation.
        assert "2000 ms" in result.explanation

    def test_result_consistent_even_without_sensor(self) -> None:
        # Robustness: no biometric data -> no crash, neutral.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=4.0,
            current_ease=2.0,
            response_time_ms=3000,
            hrv_rmssd=None,
            sleep_score=None,
        )
        assert result.bio_multiplier == pytest.approx(1.0)
        # 4 * 2.0 = 8 days standard, identical because the multiplier is neutral.
        assert result.new_interval == pytest.approx(8.0)

    def test_no_aberrant_value(self) -> None:
        # The multiplier must never be NaN nor infinite.
        result = calculate_biometric_srs(
            is_correct=True,
            previous_interval=10.0,
            current_ease=2.5,
            response_time_ms=1000,
            hrv_rmssd=60.0,
            sleep_score=75.0,
        )
        assert not math.isnan(result.bio_multiplier)
        assert not math.isinf(result.new_interval)


# ===========================================================================
# Part 2: API integration tests (isolated in-memory SQLite database)
# ===========================================================================
@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a FastAPI test client wired to an ephemeral database.

    We replace the session dependency with an in-memory SQLite database
    (`sqlite://`) shared across calls within a single test via `StaticPool`.
    Thus tests never touch the `srs_biometrics.db` file.
    """
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    def override_session() -> Session:
        session = testing_session_local()
        try:
            yield session
        finally:
            session.close()

    # We substitute the FastAPI dependency with the test session.
    main.app.dependency_overrides[main.get_db_session] = override_session

    # We insert a reference card so that we can review.
    seed_session = testing_session_local()
    seed_session.add(
        Flashcard(
            kanji="日曜日",
            kana="にちようび",
            meaning="Sunday",
            kanji_furigana="日[にち] 曜[よう] 日[び]",
            sentence_jp="日曜日は海に行きました。",
            sentence_translation="On Sunday, I went to the beach.",
            sentence_furigana="日[にち] 曜[よう] 日[び]は海に行きました。",
        )
    )
    seed_session.commit()
    seed_session.close()

    client = TestClient(main.app)
    yield client

    # Cleanup: remove the override so as not to pollute other tests.
    main.app.dependency_overrides.clear()


def test_api_health(api_client: TestClient) -> None:
    """The /health probe answers ok."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_cards_next_returns_a_card(api_client: TestClient) -> None:
    """GET /cards/next returns a complete card."""
    response = api_client.get("/cards/next")
    assert response.status_code == 200
    body = response.json()
    assert body["kanji"] == "日曜日"
    assert body["kana"] == "にちようび"


def test_api_review_contains_explanatory_fields(api_client: TestClient) -> None:
    """POST /reviews returns all requested explainability fields."""
    card = api_client.get("/cards/next").json()

    response = api_client.post(
        "/reviews",
        json={
            "flashcard_id": card["id"],
            "is_correct": True,
            "response_time_ms": 2000,
            "hrv_rmssd": 90.0,
            "sleep_score": 85.0,
        },
    )
    assert response.status_code == 201
    body = response.json()

    # Presence and consistency of the explanatory fields.
    assert "previous_interval" in body
    assert "new_interval" in body
    assert "standard_anki_interval" in body
    assert "bio_multiplier" in body
    assert "latency_category" in body
    assert "explanation" in body

    assert body["previous_interval"] == pytest.approx(0.0)
    assert body["latency_category"] == LATENCY_CATEGORY_FAST
    assert body["bio_multiplier"] > 1.0
    assert body["explanation"] != ""


def test_api_review_unknown_card_returns_404(api_client: TestClient) -> None:
    """POST /reviews on an unknown id -> explicit 404."""
    response = api_client.post(
        "/reviews",
        json={
            "flashcard_id": 99999,
            "is_correct": True,
            "response_time_ms": 2000,
            "hrv_rmssd": None,
            "sleep_score": None,
        },
    )
    assert response.status_code == 404


def test_api_review_then_telemetry_recent(api_client: TestClient) -> None:
    """After a review, /reviews/recent returns that review with the kanji."""
    card = api_client.get("/cards/next").json()
    api_client.post(
        "/reviews",
        json={
            "flashcard_id": card["id"],
            "is_correct": False,
            "response_time_ms": 7000,
            "hrv_rmssd": 40.0,
            "sleep_score": 55.0,
        },
    )

    response = api_client.get("/reviews/recent")
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["kanji"] == "日曜日"
    assert entry["is_correct"] is False
    assert entry["response_time_ms"] == 7000
    assert entry["hrv_rmssd"] == pytest.approx(40.0)
    assert entry["sleep_score"] == pytest.approx(55.0)


def test_api_reviews_recent_empty_at_start(api_client: TestClient) -> None:
    """Without reviews, the telemetry journal is an empty list."""
    response = api_client.get("/reviews/recent")
    assert response.status_code == 200
    assert response.json() == []


def test_api_cards_next_prioritises_new_card(api_client: TestClient) -> None:
    """A new card (interval 0) must be prioritised over any other."""
    # The card seeded by the fixture is new (interval 0 by default).
    response = api_client.get("/cards/next")
    assert response.status_code == 200
    body = response.json()
    assert body["interval_days"] == pytest.approx(0.0)


def test_api_cards_next_404_when_no_due_card(api_client: TestClient) -> None:
    """Once the only card is scheduled in the future, /cards/next returns 404."""
    card = api_client.get("/cards/next").json()

    # We force a long interval to push the card into the future.
    # The review sets next_review to now + interval (>= 1 day).
    response = api_client.post(
        "/reviews",
        json={
            "flashcard_id": card["id"],
            "is_correct": True,
            "response_time_ms": 2000,
            "hrv_rmssd": 60.0,
            "sleep_score": 75.0,
        },
    )
    assert response.status_code == 201

    # The card is now scheduled in the future -> nothing left to review.
    next_response = api_client.get("/cards/next")
    assert next_response.status_code == 404


def test_api_reviews_export_returns_full_history(api_client: TestClient) -> None:
    """`/reviews/export` must return all reviews, unbounded."""
    card = api_client.get("/cards/next").json()

    # We record three successive reviews.
    for correct in (True, False, True):
        api_client.post(
            "/reviews",
            json={
                "flashcard_id": card["id"],
                "is_correct": correct,
                "response_time_ms": 3000,
                "hrv_rmssd": 55.0,
                "sleep_score": 70.0,
            },
        )

    response = api_client.get("/reviews/export")
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 3
    # The sort is ascending chronological: the first is a success.
    assert entries[0]["is_correct"] is True


def test_api_reviews_export_empty_at_start(api_client: TestClient) -> None:
    """Without reviews, the export is an empty list (not an error)."""
    response = api_client.get("/reviews/export")
    assert response.status_code == 200
    assert response.json() == []


def test_timestamp_is_timezone_aware(api_client: TestClient) -> None:
    """The returned timestamps must carry a timezone (UTC).

    We verify that `reviewed_at` serialised ends with `+00:00` (or `Z`), proof
    that it was produced with `utc_now()` and is not an ambiguous naive
    datetime.
    """
    card = api_client.get("/cards/next").json()
    response = api_client.post(
        "/reviews",
        json={
            "flashcard_id": card["id"],
            "is_correct": True,
            "response_time_ms": 1000,
            "hrv_rmssd": None,
            "sleep_score": None,
        },
    )
    assert response.status_code == 201
    reviewed_at: str = response.json()["reviewed_at"]
    # An aware serialisation contains the timezone offset.
    assert reviewed_at.endswith("+00:00") or reviewed_at.endswith("Z")


def test_api_review_logs_scheduler_decision(api_client: TestClient) -> None:
    """The export holds what the scheduler decided, not only its inputs."""
    card = api_client.get("/cards/next").json()
    api_client.post(
        "/reviews",
        json={
            "flashcard_id": card["id"],
            "is_correct": True,
            "response_time_ms": 2000,
            "hrv_rmssd": 100.0,
            "sleep_score": 100.0,
        },
    )

    entry = api_client.get("/reviews/export").json()[0]
    assert entry["previous_interval"] == pytest.approx(0.0)
    assert entry["standard_interval"] == pytest.approx(1.0)
    assert entry["bio_multiplier"] == pytest.approx(1.25)
    assert entry["new_interval"] == pytest.approx(1.25)
    assert entry["new_ease"] == pytest.approx(2.55)
    assert entry["latency_category"] == LATENCY_CATEGORY_FAST


def test_create_database_adds_columns_to_old_schema(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database created before the decision columns existed gets them added."""
    from sqlalchemy import inspect, text

    from src import database

    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE review_logs (id INTEGER PRIMARY KEY, flashcard_id INTEGER, "
                "reviewed_at DATETIME, is_correct BOOLEAN, response_time_ms INTEGER, "
                "hrv_rmssd FLOAT, sleep_score FLOAT)"
            )
        )
    monkeypatch.setattr(database, "engine", engine)

    database.create_database()
    database.create_database()  # second call must be a no-op

    columns = {col["name"] for col in inspect(engine).get_columns("review_logs")}
    assert {"new_interval", "bio_multiplier", "latency_category"} <= columns
