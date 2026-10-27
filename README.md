# SRS Biometrics

[![CI](https://github.com/yanis-san/srs-research/actions/workflows/ci.yml/badge.svg)](https://github.com/yanis-san/srs-research/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A research prototype of a spaced-repetition scheduler. Besides right/wrong answers, it uses response latency, heart-rate variability (HRV) and a sleep score to adjust review intervals.

I built it to prepare a study on one question: does the learner's state at review time tell us anything about how well a word will be remembered? The prototype does not answer that question. It only makes the idea concrete enough to test.

**Current status**

- No wearable is connected. In the demo, HRV and sleep are entered by hand with two sliders.
- No study has been run. The protocol is written but not yet pre-registered, and it has no ethics approval yet.
- All constants (reference HRV and sleep values, weights, bounds, latency thresholds) are my own working assumptions, not values from the literature.

## How the scheduler works

The base is a simplified SM-2 scheduler with binary grading. A failure sends the card back to 1 day and lowers its ease by 0.2. A success multiplies the previous interval by the ease (a new card goes to 1 day).

Three signals are added on top:

- **Latency** is the time from card display to the click on "Show answer". On a success, a fast answer (< 2.5 s) raises ease by 0.05 and a slow one (> 6 s) lowers it by 0.05. Failures get no latency adjustment. Treating a fast answer as a sign of a strong memory is an assumption. Slow successes could just as well be useful "desirable difficulties", and question RQ3 below is about exactly that.
- **HRV (RMSSD)** and a **sleep score (0-100)** are combined into a multiplier on the success interval:

  ```
  m = 1 + [0.6 * (HRV - 60) / 60 + 0.4 * (sleep - 75) / 75]   clamped to [0.75, 1.25]
  ```

  If one signal is missing, its weight is dropped and the other is renormalised. With neither, m = 1.

The API also returns the interval the base scheduler would have given, so every decision can be compared with the baseline. The code is in `src/srs_logic.py`.

Limitations of the current prototype:

- Grading is self-reported ("I recalled it" / "I did not").
- The latency clock runs on the Streamlit server, so it includes a few tens of milliseconds of rendering and network delay.
- The demo shows a word, not a typed-recall test.

## Planned study

| RQ | Question | Type |
|----|----------|------|
| RQ1 | Does physiological state (HRV, sleep) predict recall at a controlled interval? | Confirmatory |
| RQ2 | Does the full scheduler reach non-inferior retention with fewer reviews than the base scheduler? | Confirmatory |
| RQ3 | Does response latency predict forgetting beyond item difficulty? | Confirmatory |
| RQ4 | Which constants matter, and what values does the data support? | Exploratory |
| RQ5 | Would an adaptive (Bayesian or bandit) policy do better than a fixed multiplier? | Future work |

The planned design is within-subject. Items are randomly assigned to four versions of the same base scheduler: base only, + latency, + HRV/sleep, and all three. Each item is tested once, at +7 or +30 days, with typed cued recall. The analysis would use mixed-effects logistic regression with participant and item random effects. The sample size is to be set by a simulation-based power analysis on pilot data; the provisional target is about 40 analysable participants.

The full protocol is a separate document and is not in this repository.

## Sensitivity analysis (RQ4)

Each constant is moved by ±20 %, one at a time, over six hand-written scenarios. Three things are recorded: the interval shift, the effect over five successive reviews, and, for the latency thresholds, how many of 30 hand-picked reference latencies change category.

At ±20 %:

- The multiplier bounds and the HRV and sleep reference values shift the interval the most (up to ±6.25 days for the upper bound). This is partly because two of the scenarios are extreme and hit the bounds.
- Moving a latency threshold reclassifies 2 of the 30 reference latencies (fast threshold, 6.7 %) or 4 of 30 (slow threshold, 13.3 %).
- The 0.6/0.4 weights barely matter (about ±0.008 days). The ease constants only act on later reviews and stay under 2 % over five reviews.
- The mean Spearman ρ between nominal and perturbed intervals is 0.995.

None of this says whether the constants are right. A bounded, monotonic multiplier is expected to keep the ranking stable, and ρ over six scenarios (two of which tie at 1 day) is only a coarse check. A global method (Morris or Sobol) on real data would be the next step.

```bash
uv run python scripts/sensitivity_analysis.py --perturbation 0.2
```

## Running it

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run etl.py                              # imports data/demo.tsv (15 cards)
uv run uvicorn src.main:app --reload       # API, docs at http://127.0.0.1:8000/docs
uv run streamlit run src/app.py            # dashboard, in a second terminal
```

### Using a full Anki deck

No third-party deck is included. `data/demo.tsv` is a small Japanese list I wrote for the demo. To use a real deck, export it from Anki as UTF-8 tab-separated text and either save it as `data/kaishi.txt` (picked up automatically) or run `uv run etl.py path/to/deck.txt`. The expected columns are listed in `etl.py`. Cards are de-duplicated on the first column, so the import can be re-run safely.

## API

| Method | Route | |
|--------|-------|---|
| GET | `/health` | Health check |
| GET | `/cards/next` | Next due card (new cards first), 404 if none |
| POST | `/reviews` | Record a review; returns the new interval, the baseline interval and a one-line explanation |
| GET | `/reviews/recent` | Last 10 reviews |
| GET | `/reviews/export` | All reviews, with inputs and the scheduler's decision |
| GET | `/stats` | Card counts, success rate, mean latency |

```bash
curl -X POST http://127.0.0.1:8000/reviews \
  -H "Content-Type: application/json" \
  -d '{"flashcard_id": 1, "is_correct": true, "response_time_ms": 2000, "hrv_rmssd": 75, "sleep_score": 85}'
```

## Code layout

```
src/
  srs_logic.py      scheduling rules (pure functions)
  sensitivity.py    sensitivity analysis
  database.py       SQLAlchemy models, SQLite
  schemas.py        Pydantic models
  main.py           FastAPI app
  app.py            Streamlit dashboard
  config.py         settings (SRS_* environment variables)
scripts/sensitivity_analysis.py
etl.py              deck import
tests/              pytest
```

Stack: Python 3.11+, FastAPI, SQLAlchemy 2.0 with SQLite, Pydantic v2, Streamlit, pytest, uv.

## Tests

```bash
uv run pytest -v
```

There are 64 tests: 45 cover the scheduling rules and the API, and 19 cover the sensitivity analysis. CI runs them on Python 3.11, 3.12, 3.13 and 3.14.

## Not implemented yet

These are needed before the study could run:

- per-item assignment to the four conditions, hidden from the participant;
- a personal rolling HRV baseline instead of the fixed 60 ms reference;
- logging raw sleep components (duration, efficiency) and sleep after the study session;
- an explicit missing-sensor flag instead of a silent neutral multiplier;
- a typed-recall test mode with automatic scoring;
- offline replay of the logged reviews against FSRS.

## Data and license

No personal data is in this repository. The local database (`srs_biometrics.db`) and `.env` are git-ignored. This is not a medical device and makes no health claims.

MIT license, see `LICENSE`.
