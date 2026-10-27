"""Import cards from a tab-separated Anki export into the SQLite database.

    uv run etl.py              # data/kaishi.txt if present, else data/demo.tsv
    uv run etl.py path/to/deck.txt

Cards are de-duplicated on the kanji field, so the script can be re-run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select  # noqa: E402

from src.database import Flashcard, create_database, get_session  # noqa: E402

DECK_FILE: Path = PROJECT_ROOT / "data" / "kaishi.txt"
DEMO_FILE: Path = PROJECT_ROOT / "data" / "demo.tsv"

# Columns used: 0-3, 5-7 and optionally 9 (see parse_line).
MIN_COLUMNS: int = 8


def clean_text(text: str | None) -> str:
    """Strip Anki markup but keep furigana brackets such as 日[にち].

    Order matters: [sound:...] goes first because it also uses brackets.
    <span> blocks are removed with their content (pitch-accent markup).
    """
    if text is None:
        return ""
    text = re.sub(r"\[sound:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<span[^>]*>.*?</span>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Only tags that start with a letter, so a stray "<" in text survives.
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    # Anki wraps some fields in quotes.
    text = re.sub(r'^["\']+|["\']+$', "", text)
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_line(line: str) -> dict[str, str] | None:
    """Map one TSV line to card fields, or None if the line is unusable.

    Columns: 0 kanji, 1 kana, 2 meaning, 3 word furigana, 4 word audio
    (ignored), 5 sentence, 6 sentence translation, 7 sentence furigana,
    8 sentence audio (ignored), 9 notes (optional).
    """
    stripped = line.strip()
    # Empty lines and Anki headers such as "#separator:tab".
    if not stripped or stripped.startswith("#"):
        return None

    columns = line.split("\t")
    if len(columns) < MIN_COLUMNS:
        return None

    kanji = clean_text(columns[0])
    if not kanji:
        return None

    return {
        "kanji": kanji,
        "kana": clean_text(columns[1]),
        "meaning": clean_text(columns[2]),
        "kanji_furigana": clean_text(columns[3]),
        "sentence_jp": clean_text(columns[5]),
        "sentence_translation": clean_text(columns[6]),
        "sentence_furigana": clean_text(columns[7]),
        "notes": clean_text(columns[9]) if len(columns) > 9 else "",
    }


def pick_source(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    return DECK_FILE if DECK_FILE.exists() else DEMO_FILE


def load_cards(source: Path) -> None:
    if not source.exists():
        print(f"File not found: {source}")
        return

    print(f"Importing {source}")
    create_database()
    raw_lines = source.read_text(encoding="utf-8").splitlines()

    inserted = 0
    duplicates = 0
    invalid = 0

    session = get_session()
    try:
        existing = set(session.scalars(select(Flashcard.kanji)).all())

        for line in raw_lines:
            data = parse_line(line)
            if data is None:
                invalid += 1
                continue
            if data["kanji"] in existing:
                duplicates += 1
                continue

            session.add(
                Flashcard(
                    kanji=data["kanji"],
                    kana=data["kana"],
                    meaning=data["meaning"],
                    kanji_furigana=data["kanji_furigana"],
                    sentence_jp=data["sentence_jp"],
                    sentence_translation=data["sentence_translation"],
                    sentence_furigana=data["sentence_furigana"],
                    notes=data["notes"] or None,
                )
            )
            existing.add(data["kanji"])
            inserted += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Lines read: {len(raw_lines)}")
    print(f"Inserted: {inserted}")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Invalid lines skipped: {invalid}")


if __name__ == "__main__":
    load_cards(pick_source(sys.argv))
