"""Streamlit dashboard for the prototype.

Run the API first (uv run uvicorn src.main:app --reload), then:
    uv run streamlit run src/app.py

No wearable is connected: HRV and sleep come from two sliders in the sidebar.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# `streamlit run` does not put the project root on sys.path.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.sensitivity import run_sensitivity_analysis  # noqa: E402
from src.srs_logic import (  # noqa: E402
    MAX_BIO_MULTIPLIER,
    MIN_BIO_MULTIPLIER,
    NEUTRAL_HRV_RMSSD,
    NEUTRAL_SLEEP_SCORE,
    calculate_bio_multiplier,
    classify_latency,
)

API_BASE_URL: str = settings.api_base_url
REQUEST_TIMEOUT_SECONDS: int = 5

SESSION_CARD_KEY: str = "current_card"
SESSION_START_TIME_KEY: str = "card_start_time"
SESSION_LATENCY_KEY: str = "latency_ms"
SESSION_REVEAL_KEY: str = "reveal"
SESSION_LAST_FEEDBACK_KEY: str = "last_feedback"


def _get(path: str) -> requests.Response | None:
    try:
        return requests.get(f"{API_BASE_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.ConnectionError:
        return None


def fetch_next_card() -> dict | None:
    """Next due card, or None. A 404 just means nothing is due."""
    response = _get("/cards/next")
    if response is None:
        st.error("Cannot reach the API. Start it with `uv run uvicorn src.main:app --reload`.")
        return None
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        st.error(f"API error ({response.status_code}): {response.text}")
        return None
    return response.json()


def fetch_json(path: str) -> dict | list | None:
    response = _get(path)
    if response is None or response.status_code != 200:
        return None
    return response.json()


def load_card_into_session() -> None:
    """Load the next card and start the latency clock."""
    card = fetch_next_card()
    st.session_state[SESSION_CARD_KEY] = card
    st.session_state[SESSION_REVEAL_KEY] = False
    st.session_state[SESSION_LATENCY_KEY] = None
    # Started when the front is rendered. The clock runs server-side, so the
    # value includes page rendering and network delay (tens of ms).
    st.session_state[SESSION_START_TIME_KEY] = time.perf_counter() if card else None


def stop_latency_clock() -> None:
    """Latency = card display to 'Show answer' click.

    Stopping here keeps the time spent reading the back of the card out of
    the measure.
    """
    start = st.session_state.get(SESSION_START_TIME_KEY)
    if start is not None:
        st.session_state[SESSION_LATENCY_KEY] = int((time.perf_counter() - start) * 1000)


def send_review(is_correct: bool, hrv_rmssd: float, sleep_score: float) -> None:
    card = st.session_state.get(SESSION_CARD_KEY)
    latency_ms = st.session_state.get(SESSION_LATENCY_KEY)
    if card is None or latency_ms is None:
        st.error("No latency recorded for this card; reload the page.")
        return

    payload = {
        "flashcard_id": card["id"],
        "is_correct": is_correct,
        "response_time_ms": latency_ms,
        "hrv_rmssd": hrv_rmssd,
        "sleep_score": sleep_score,
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/reviews",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        st.error("Lost connection to the API.")
        return

    if response.status_code != 201:
        st.error(f"Could not record the review: {response.text}")
        return

    feedback = response.json()
    feedback["reviewed_kanji"] = card.get("kanji", "?")
    st.session_state[SESSION_LAST_FEEDBACK_KEY] = feedback
    load_card_into_session()


def render_sidebar() -> tuple[float, float]:
    st.sidebar.header("HRV and sleep (manual input)")
    st.sidebar.caption("No device is connected. Set the values by hand to see their effect.")

    hrv_rmssd = st.sidebar.slider("HRV, RMSSD (ms)", min_value=20, max_value=100, value=65, step=1)
    sleep_score = st.sidebar.slider("Sleep score (0-100)", min_value=40, max_value=100, value=80, step=1)

    multiplier = calculate_bio_multiplier(hrv_rmssd, sleep_score)
    st.sidebar.markdown(f"**Interval multiplier: x{multiplier:.2f}**")
    span = MAX_BIO_MULTIPLIER - MIN_BIO_MULTIPLIER
    st.sidebar.progress(max(0.0, min(1.0, (multiplier - MIN_BIO_MULTIPLIER) / span)))
    st.sidebar.caption(
        f"1.00 at HRV {NEUTRAL_HRV_RMSSD:.0f} ms and sleep {NEUTRAL_SLEEP_SCORE:.0f}/100; "
        f"bounded to [{MIN_BIO_MULTIPLIER}, {MAX_BIO_MULTIPLIER}]. "
        "These reference values are assumptions, not norms."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Statistics")
    stats = fetch_json("/stats")
    if stats is None:
        st.sidebar.info("API offline.")
    else:
        st.sidebar.metric("Cards", stats["total_cards"])
        st.sidebar.metric("Due", stats["due_cards"])
        st.sidebar.metric("Success rate", f"{stats['success_rate'] * 100:.1f} %")
        st.sidebar.metric("Mean latency", f"{stats['average_response_time_ms']:.0f} ms")

    return hrv_rmssd, sleep_score


def render_last_feedback() -> None:
    feedback = st.session_state.get(SESSION_LAST_FEEDBACK_KEY)
    if feedback is None:
        return

    outcome = "recalled" if feedback.get("is_correct") else "not recalled"
    st.subheader(f"Last card: {feedback.get('reviewed_kanji', '?')} ({outcome})")

    col_latency, col_multiplier, col_interval = st.columns(3)
    col_latency.metric("Latency", f"{feedback['response_time_ms']} ms", feedback["latency_category"])
    col_multiplier.metric("Multiplier", f"x{feedback['bio_multiplier']:.2f}")
    col_interval.metric(
        "New interval",
        f"{feedback['new_interval']:.1f} d",
        delta=(
            f"{feedback['new_interval'] - feedback['standard_anki_interval']:+.1f} d "
            f"vs base ({feedback['standard_anki_interval']:.1f} d)"
        ),
    )
    st.info(feedback["explanation"])


def render_sensitivity_analysis() -> None:
    st.subheader("Sensitivity of the constants (RQ4)")
    st.caption(
        "Each constant is moved by the chosen amount, one at a time, on six "
        "hand-written scenarios. This shows which constants move the schedule; "
        "it does not show that their values are right."
    )
    perturbation_pct = st.slider("Perturbation (+/- %)", min_value=5, max_value=50, value=20, step=5)

    if st.button("Run"):
        report = run_sensitivity_analysis(perturbation=perturbation_pct / 100.0)
        st.write(report.summary)
        st.dataframe(
            [
                {
                    "Constant": f.name,
                    "Nominal": round(f.nominal, 3),
                    "Mean delta + (d)": round(f.mean_interval_delta_plus, 3),
                    "Mean delta - (d)": round(f.mean_interval_delta_minus, 3),
                    "Max |delta| (d)": round(f.max_abs_delta, 3),
                    "Elasticity": round(f.mean_elasticity, 3),
                    "Trajectory": round(f.timeline_impact, 3),
                    "Reclassified (%)": round(f.reclassification_rate * 100, 1),
                    "Influential": "yes" if f.is_influential else "no",
                }
                for f in report.factors
            ],
            width="stretch",
            hide_index=True,
        )


def render_research_analytics() -> None:
    st.subheader("Review data")

    history = fetch_json("/reviews/export")
    if not history:
        st.info("No reviews yet.")
        return

    df = pd.DataFrame(history)

    st.markdown("**Latency distribution (1 s bins)**")
    seconds = (df["response_time_ms"] // 1000).astype(int)
    counts = seconds.value_counts().sort_index()
    counts.index = [f"{s}-{s + 1} s" for s in counts.index]
    st.bar_chart(counts)

    st.markdown("**Self-graded success rate by latency category**")
    df["category"] = df["response_time_ms"].apply(classify_latency)
    st.bar_chart(df.groupby("category")["is_correct"].mean())

    st.download_button(
        label="Download review_logs.csv",
        data=df.drop(columns="category").to_csv(index=False).encode("utf-8"),
        file_name="review_logs.csv",
        mime="text/csv",
    )


def render_recent_reviews() -> None:
    with st.expander("Last 10 reviews"):
        recent = fetch_json("/reviews/recent")
        if not recent:
            st.write("No reviews yet.")
            return
        st.dataframe(
            [
                {
                    "ID": entry["id"],
                    "Card": entry["kanji"],
                    "Recalled": "yes" if entry["is_correct"] else "no",
                    "Latency (ms)": entry["response_time_ms"],
                    "HRV (ms)": entry["hrv_rmssd"],
                    "Sleep": entry["sleep_score"],
                    "Multiplier": entry["bio_multiplier"],
                    "New interval (d)": entry["new_interval"],
                    "Time (UTC)": entry["reviewed_at"],
                }
                for entry in recent
            ],
            width="stretch",
            hide_index=True,
        )


def render_footer() -> None:
    st.divider()
    render_research_analytics()
    st.divider()
    render_sensitivity_analysis()
    st.divider()
    render_recent_reviews()


def main() -> None:
    st.set_page_config(page_title="SRS Biometrics", layout="centered")
    st.title("SRS Biometrics prototype")
    st.caption("Spaced repetition with intervals adjusted by response latency, HRV and sleep.")

    hrv_rmssd, sleep_score = render_sidebar()
    render_last_feedback()

    if SESSION_CARD_KEY not in st.session_state:
        load_card_into_session()

    card = st.session_state.get(SESSION_CARD_KEY)
    if card is None:
        st.success("No card is due right now.")
        if st.button("Check again"):
            load_card_into_session()
            st.rerun()
        render_footer()
        return

    st.divider()
    st.markdown(
        f"<h1 style='text-align: center; font-size: 84px;'>{card['kanji']}</h1>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get(SESSION_REVEAL_KEY, False):
        st.caption("Try to recall the reading and meaning, then show the answer.")
        if st.button("Show answer", width="stretch"):
            stop_latency_clock()
            st.session_state[SESSION_REVEAL_KEY] = True
            st.rerun()
    else:
        st.divider()
        st.markdown(f"**Reading:** {card['kana']}")
        st.markdown(f"**Furigana:** {card['kanji_furigana']}")
        st.markdown(f"**Meaning:** {card['meaning']}")
        st.markdown("---")
        st.markdown(f"**Sentence:** {card['sentence_jp']}")
        st.markdown(f"**Furigana:** {card['sentence_furigana']}")
        st.markdown(f"**Translation:** {card['sentence_translation']}")
        if card.get("notes"):
            st.info(card["notes"])

        st.divider()
        st.caption(f"Latency recorded: {st.session_state.get(SESSION_LATENCY_KEY)} ms")
        col_yes, col_no = st.columns(2)
        if col_yes.button("I recalled it", width="stretch"):
            send_review(True, hrv_rmssd, sleep_score)
            st.rerun()
        if col_no.button("I did not recall it", width="stretch"):
            send_review(False, hrv_rmssd, sleep_score)
            st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
