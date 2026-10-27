"""SQLite engine and ORM models (cards and review logs)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import ForeignKey, create_engine, inspect, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import DateTime, TypeDecorator

from src.config import settings
from src.time_utils import utc_now

DATABASE_URL: str = settings.database_url

# FastAPI runs handlers in a thread pool, hence check_same_thread=False.
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class UtcDateTime(TypeDecorator):
    """Stores UTC datetimes and always returns them timezone-aware.

    SQLite drops the timezone, so it is re-attached on read. Naive values
    written by the application are assumed to be UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class Flashcard(Base):
    __tablename__ = "flashcards"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Indexed because the ETL de-duplicates on it.
    kanji: Mapped[str] = mapped_column(index=True)
    kana: Mapped[str]
    # The column names keep their original "_fr" suffix so existing local
    # databases still load; the content can be in any language.
    meaning: Mapped[str] = mapped_column("meaning_fr")
    kanji_furigana: Mapped[str]
    sentence_jp: Mapped[str]
    sentence_translation: Mapped[str] = mapped_column("sentence_fr")
    sentence_furigana: Mapped[str]
    notes: Mapped[str | None] = mapped_column(default=None)

    # Scheduler state. interval_days == 0 means the card has never been reviewed.
    interval_days: Mapped[float] = mapped_column(default=0.0)
    ease_factor: Mapped[float] = mapped_column(default=2.5)
    next_review: Mapped[datetime] = mapped_column(UtcDateTime, default=utc_now, index=True)

    review_logs: Mapped[list[ReviewLog]] = relationship(
        back_populates="flashcard",
        cascade="all, delete-orphan",
    )


class ReviewLog(Base):
    """One review: what was measured and what the scheduler decided."""

    __tablename__ = "review_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    flashcard_id: Mapped[int] = mapped_column(ForeignKey("flashcards.id"), index=True)
    reviewed_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utc_now, index=True)

    # Inputs. is_correct is self-graded by the learner during reviews.
    is_correct: Mapped[bool]
    response_time_ms: Mapped[int]
    hrv_rmssd: Mapped[float | None] = mapped_column(default=None)
    sleep_score: Mapped[float | None] = mapped_column(default=None)

    # Scheduler decision. Nullable because rows logged before these columns
    # existed do not have them.
    previous_interval: Mapped[float | None] = mapped_column(default=None)
    new_interval: Mapped[float | None] = mapped_column(default=None)
    standard_interval: Mapped[float | None] = mapped_column(default=None)
    new_ease: Mapped[float | None] = mapped_column(default=None)
    bio_multiplier: Mapped[float | None] = mapped_column(default=None)
    latency_category: Mapped[str | None] = mapped_column(default=None)

    flashcard: Mapped[Flashcard] = relationship(back_populates="review_logs")


# Columns added to review_logs after the first version of the schema.
_ADDED_REVIEW_LOG_COLUMNS: dict[str, str] = {
    "previous_interval": "FLOAT",
    "new_interval": "FLOAT",
    "standard_interval": "FLOAT",
    "new_ease": "FLOAT",
    "bio_multiplier": "FLOAT",
    "latency_category": "VARCHAR",
}


def _add_missing_columns() -> None:
    """Add new nullable columns to a database created by an older version."""
    existing = {col["name"] for col in inspect(engine).get_columns("review_logs")}
    with engine.begin() as conn:
        for name, sql_type in _ADDED_REVIEW_LOG_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE review_logs ADD COLUMN {name} {sql_type}"))


def create_database() -> None:
    """Create missing tables and columns. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def get_session() -> Session:
    """New session; the caller closes it."""
    return SessionLocal()
