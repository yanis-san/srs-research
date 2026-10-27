"""Request and response models of the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FlashcardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kanji: str
    kana: str
    meaning: str
    kanji_furigana: str
    sentence_jp: str
    sentence_translation: str
    sentence_furigana: str
    notes: str | None = None

    interval_days: float
    ease_factor: float
    next_review: datetime


class ReviewCreate(BaseModel):
    flashcard_id: int = Field(..., gt=0)
    is_correct: bool = Field(..., description="Self-graded: recalled or not")
    response_time_ms: int = Field(
        ...,
        ge=0,
        description="Time from card display to the 'Show answer' click, in ms",
    )
    hrv_rmssd: float | None = Field(default=None, ge=0, description="RMSSD in ms")
    sleep_score: float | None = Field(default=None, ge=0, le=100)


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    flashcard_id: int
    reviewed_at: datetime
    is_correct: bool
    response_time_ms: int
    hrv_rmssd: float | None = None
    sleep_score: float | None = None

    previous_interval: float = Field(..., description="Interval before this review (days)")
    new_interval: float = Field(..., description="Interval after this review (days)")
    standard_anki_interval: float = Field(
        ...,
        description="Interval the base scheduler would give without latency or HRV/sleep",
    )
    bio_multiplier: float = Field(..., description="HRV/sleep multiplier, in [0.75, 1.25]")
    latency_category: str
    explanation: str


class TelemetryEntry(BaseModel):
    """One row of /reviews/recent and /reviews/export."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    flashcard_id: int
    kanji: str
    reviewed_at: datetime
    is_correct: bool
    response_time_ms: int
    hrv_rmssd: float | None = None
    sleep_score: float | None = None
    # Scheduler decision; null for reviews logged by older versions.
    previous_interval: float | None = None
    new_interval: float | None = None
    standard_interval: float | None = None
    new_ease: float | None = None
    bio_multiplier: float | None = None
    latency_category: str | None = None


class StatsResponse(BaseModel):
    total_cards: int
    due_cards: int
    success_rate: float = Field(..., ge=0, le=1)
    average_response_time_ms: float = Field(..., ge=0)
