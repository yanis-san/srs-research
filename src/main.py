"""FastAPI app.

Run with: uv run uvicorn src.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database import Flashcard, ReviewLog, create_database, get_session
from src.schemas import (
    FlashcardResponse,
    ReviewCreate,
    ReviewResponse,
    StatsResponse,
    TelemetryEntry,
)
from src.srs_logic import calculate_biometric_srs
from src.time_utils import utc_now

RECENT_REVIEWS_LIMIT: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    create_database()
    yield


app = FastAPI(
    title="SRS Biometrics API",
    description=(
        "Research prototype: spaced repetition with intervals modulated by "
        "response latency, HRV and sleep."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_db_session() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@app.get("/cards/next", response_model=FlashcardResponse)
def get_next_card(session: Session = Depends(get_db_session)) -> Flashcard:
    """New cards first, then the most overdue card. 404 if nothing is due."""
    new_card = session.scalars(
        select(Flashcard)
        .where(Flashcard.interval_days == 0)
        .order_by(Flashcard.id.asc())
        .limit(1)
    ).first()
    if new_card is not None:
        return new_card

    due_card = session.scalars(
        select(Flashcard)
        .where(Flashcard.next_review <= utc_now())
        .order_by(Flashcard.next_review.asc())
        .limit(1)
    ).first()
    if due_card is not None:
        return due_card

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No card is due right now.",
    )


@app.post("/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
def create_review(
    review: ReviewCreate,
    session: Session = Depends(get_db_session),
) -> ReviewResponse:
    card = session.get(Flashcard, review.flashcard_id)
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flashcard id={review.flashcard_id} not found.",
        )

    previous_interval = card.interval_days
    result = calculate_biometric_srs(
        is_correct=review.is_correct,
        previous_interval=previous_interval,
        current_ease=card.ease_factor,
        response_time_ms=review.response_time_ms,
        hrv_rmssd=review.hrv_rmssd,
        sleep_score=review.sleep_score,
    )

    card.interval_days = result.new_interval
    card.ease_factor = result.new_ease
    card.next_review = utc_now() + timedelta(days=result.new_interval)

    review_log = ReviewLog(
        flashcard_id=card.id,
        is_correct=review.is_correct,
        response_time_ms=review.response_time_ms,
        hrv_rmssd=review.hrv_rmssd,
        sleep_score=review.sleep_score,
        previous_interval=previous_interval,
        new_interval=result.new_interval,
        standard_interval=result.standard_anki_interval,
        new_ease=result.new_ease,
        bio_multiplier=result.bio_multiplier,
        latency_category=result.latency_category,
    )
    session.add(review_log)
    session.commit()
    session.refresh(review_log)

    return ReviewResponse(
        id=review_log.id,
        flashcard_id=review_log.flashcard_id,
        reviewed_at=review_log.reviewed_at,
        is_correct=review_log.is_correct,
        response_time_ms=review_log.response_time_ms,
        hrv_rmssd=review_log.hrv_rmssd,
        sleep_score=review_log.sleep_score,
        previous_interval=round(previous_interval, 2),
        new_interval=result.new_interval,
        standard_anki_interval=result.standard_anki_interval,
        bio_multiplier=result.bio_multiplier,
        latency_category=result.latency_category,
        explanation=result.explanation,
    )


def _telemetry_query():
    return select(
        ReviewLog.id,
        ReviewLog.flashcard_id,
        Flashcard.kanji,
        ReviewLog.reviewed_at,
        ReviewLog.is_correct,
        ReviewLog.response_time_ms,
        ReviewLog.hrv_rmssd,
        ReviewLog.sleep_score,
        ReviewLog.previous_interval,
        ReviewLog.new_interval,
        ReviewLog.standard_interval,
        ReviewLog.new_ease,
        ReviewLog.bio_multiplier,
        ReviewLog.latency_category,
    ).join(Flashcard, Flashcard.id == ReviewLog.flashcard_id)


@app.get("/reviews/recent", response_model=list[TelemetryEntry])
def get_recent_reviews(session: Session = Depends(get_db_session)) -> list[TelemetryEntry]:
    """Last 10 reviews, newest first."""
    statement = (
        _telemetry_query()
        .order_by(ReviewLog.reviewed_at.desc(), ReviewLog.id.desc())
        .limit(RECENT_REVIEWS_LIMIT)
    )
    rows = session.execute(statement).all()
    return [TelemetryEntry.model_validate(row._mapping) for row in rows]


@app.get("/reviews/export", response_model=list[TelemetryEntry])
def export_reviews(session: Session = Depends(get_db_session)) -> list[TelemetryEntry]:
    """Full review history, oldest first."""
    statement = _telemetry_query().order_by(ReviewLog.reviewed_at.asc(), ReviewLog.id.asc())
    rows = session.execute(statement).all()
    return [TelemetryEntry.model_validate(row._mapping) for row in rows]


@app.get("/stats", response_model=StatsResponse)
def get_stats(session: Session = Depends(get_db_session)) -> StatsResponse:
    total_cards = session.scalar(select(func.count()).select_from(Flashcard)) or 0
    due_cards = session.scalar(
        select(func.count()).select_from(Flashcard).where(Flashcard.next_review <= utc_now())
    ) or 0
    total_reviews = session.scalar(select(func.count()).select_from(ReviewLog)) or 0

    if total_reviews == 0:
        return StatsResponse(
            total_cards=total_cards,
            due_cards=due_cards,
            success_rate=0.0,
            average_response_time_ms=0.0,
        )

    correct_reviews = session.scalar(
        select(func.count()).select_from(ReviewLog).where(ReviewLog.is_correct.is_(True))
    ) or 0
    average_response_time = session.scalar(select(func.avg(ReviewLog.response_time_ms))) or 0.0

    return StatsResponse(
        total_cards=total_cards,
        due_cards=due_cards,
        success_rate=round(correct_reviews / total_reviews, 4),
        average_response_time_ms=round(float(average_response_time), 2),
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
