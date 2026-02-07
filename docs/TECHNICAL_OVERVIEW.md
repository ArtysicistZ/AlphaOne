# alphaone Technical Overview

This document records the technical structure, components, and data flow of the `alphaone` codebase. It is intended as an internal engineering reference.

## Repository Layout

- `backend/`: Python backend for ingestion, NLP processing, data storage, and API serving.
- `frontend/`: React + Vite single-page application.
- `.env`: Local configuration for database and Reddit API credentials (contains secrets).
- `README.md`: Empty at root.

## Runtime Architecture (Current)

- **Data ingestion**: Scripts in `backend/core/` fetch Reddit data and store raw files in `backend/data/reddit/raw`.
- **Batch processing**: `backend/batch_processor.py` reads raw files, performs NLP (sentence-level), tags topics, stores sentiment and word frequency in PostgreSQL.
- **API**: FastAPI in `backend/api/main.py` serves data for frontend consumption.
- **Frontend**: React SPA in `frontend/` calls API endpoints and renders sentiment summaries + word cloud.

## Backend Details

### Configuration

- `backend/config.py` loads environment variables from `.env` and exposes:
  - `DATABASE_URL`
  - `REDDIT_CLIENT_ID`
  - `REDDIT_SECRET_KEY`
  - `REDDIT_USERNAME`
  - `REDDIT_PASSWORD`

The `.env` file contains secrets and should **not** be committed. Values are currently stored in plaintext and must be rotated before any public release.

### Database

File: `backend/db/models.py`

**Tables (SQLAlchemy ORM):**

- `topics`
  - `id` (PK)
  - `slug` (unique, indexed) — ticker or topic key (e.g., `NVDA`, `MACRO`)
  - `name` — display name
- `sentiment_data`
  - `id` (PK)
  - `created_at` (timestamp, default `now()`)
  - `source_id` (unique)
  - `source_type` (string)
  - `sentiment_label` (`positive`/`negative`/`neutral`)
  - `sentiment_score` (float; positive - negative probability)
  - `relevant_text` (nullable string; sentence-level evidence)
- `word_frequency`
  - `id` (PK)
  - `word` (indexed)
  - `frequency`
  - `date` (default `current_date()`)
  - Unique constraint on (`word`, `date`)

**Relationships:**

- `sentiment_data` <-> `topics` is many-to-many via `sentiment_topic_association`.

File: `backend/db/database.py`

- Creates engine using `DATABASE_URL`
- Enables `pool_pre_ping=True` (useful for Neon/serverless connections)
- Provides `SessionLocal`, `init_db()`, and `get_db_session()` dependency

### NLP & Topic Tagging

File: `backend/core/nlp_processor.py`

- Loads FinBERT (`ProsusAI/finbert`) via `transformers`
- Computes softmax probabilities and score = `P(pos) - P(neg)`
- Label mapping:
  - `score > 0.05` → `positive`
  - `score < -0.05` → `negative`
  - otherwise `neutral`

File: `backend/core/text_utils.py`

- Uses spaCy `en_core_web_md`
- `_pre_clean_text` strips URLs and markdown while preserving sentence boundaries
- `split_into_sentences` returns filtered sentences (length >= 15)

File: `backend/core/sentiment_tagger/topic_definitions.py`

- `SENTENCE_TOPIC_MAP` maps tickers/topics to keyword lists (e.g., `NVDA`, `AAPL`, `MACRO`, `TECHNOLOGY`)

File: `backend/core/sentiment_tagger/tagger_logic.py`

- `get_topics_from_sentence(sentence)` returns a set of topic slugs found by keyword substring matching

### Word Cloud

Files: `backend/core/wordcloud/*`

- `word_counter.py`: tokenizes text, removes stop words, weights keyword hits via `keyword_map.py`
- `stop_words.py`: hardcoded stop-word set
  - Note: missing comma between `"reddit"` and `"stock"` results in the unintended token `"redditstock"`
- `keyword_map.py`: assigns weights to company/ticker/macro keywords

### Data Ingestion

File: `backend/core/reddit_client.py`

- Uses PRAW to authenticate and fetch Reddit data.
- Provides:
  - `fetch_comments_from_subreddit(...)`
  - `fetch_submissions_from_subreddit(...)`
  - `search_subreddit(...)`
- Saves to `backend/data/reddit/raw` as CSV/JSON.

### Batch Processing

File: `backend/batch_processor.py`

- Reads raw files (CSV/JSON)
- For each record:
  - Updates word cloud counters (full text)
  - Splits text into sentences
  - Tags topics per sentence
  - Runs FinBERT sentiment per sentence
  - Inserts `SentimentData` with `relevant_text` = sentence
  - Creates/links `Topic` rows
- After all files:
  - Writes top 100 words into `word_frequency` for today

### API Server

File: `backend/api/main.py`

- FastAPI app with CORS allowed for:
  - `http://localhost:5173`
  - `https://alphaone.run.place`
- Routers mounted:
  - `assets` at `/api/v1/assets`
  - `social_sentiment` at `/api/v1/signals/social-sentiment`

File: `backend/api/routers/assets.py`

- `GET /api/v1/assets/tracked`
  - Returns all `Topic` rows

File: `backend/api/routers/social_sentiment.py`

- `GET /api/v1/signals/social-sentiment/{ticker}/daily`
  - Daily avg sentiment for ticker (grouped by day)
- `GET /api/v1/signals/social-sentiment/{ticker}/evidence`
  - Last 5 sentiment records for ticker (sentence evidence)
- `GET /api/v1/signals/social-sentiment/summary/{topic_slug}`
  - Average sentiment across all records for topic
- `GET /api/v1/signals/social-sentiment/wordcloud`
  - Top 100 words (by frequency)

### Maintenance Script

File: `backend/db_cleanup.py`

- Deletes sentiment and word frequency rows older than 7 days

### Legacy / Stale Code

File: `backend/worker.py`

- References `core.topic_tagger` and `SentimentData.text_content` which do not exist.
- `get_sentiment` signature is inconsistent (expects dict but current returns tuple).
- Treat as outdated and not part of the current pipeline.

### Backend Dependencies (Inferred)

`backend/requirements.txt` is empty; dependencies are inferred from imports:

- `fastapi`, `uvicorn`
- `sqlalchemy`
- `python-dotenv`
- `praw`
- `pandas`
- `transformers`, `torch`, `numpy`
- `spacy` (and `en_core_web_md` model)

## Frontend Details

### Tooling

File: `frontend/package.json`

- Vite + React
- Dependencies:
  - `react`, `react-dom`, `react-router-dom`
  - `axios`
  - `chart.js`, `react-chartjs-2`
  - `react-d3-cloud`

### Entry + Routing

File: `frontend/src/main.jsx`

- Wraps app with `BrowserRouter`.

File: `frontend/src/App.jsx`

- Parent route with layout wrapper:
  - `/` → `HomePage`
  - `/sentiment-summary` → `SentimentSummaryPage`
- Other routes are placeholders only.

### Layout

Files:

- `frontend/src/components/Layout.jsx`: renders `Navbar`, `Outlet`, `Footer`
- `frontend/src/components/Navbar.jsx`: top navigation bar
- `frontend/src/components/Footer.jsx`: footer with external image link

### Pages

- `HomePage`: simple hero text.
- `SentimentSummaryPage`: main UI
  - Calls API for:
    - `MACRO` and `TECHNOLOGY` summary scores
    - Word cloud data
    - Tracked topics (for autocomplete)
  - Search form:
    - Fetches summary + evidence for a ticker
  - Renders word cloud with `react-d3-cloud`
- `DashboardPage` exists but is not routed in `App.jsx`.

### API Client

Files:

- `frontend/src/api/index.js`: Axios client with base URL = `VITE_API_URL` or `http://127.0.0.1:8000`.
- `frontend/src/api/sentimentApi.js`: wraps endpoints for summary, evidence, word cloud, and tracked topics.

### Styling

- `frontend/src/index.css`: basic reset + `#root` flex layout.
- Many components use inline styles (no CSS framework).

## Data Files

Example raw data (JSON) exists in:

- `backend/data/reddit/raw/*`

These are used by `batch_processor.py`.

## Notable Technical Risks / Gaps

- Secrets committed in `.env` (must rotate + remove from VCS).
- `backend/requirements.txt` is empty; environment setup is undefined.
- `backend/worker.py` is stale and inconsistent with current models.
- Stop-word list bug in `backend/core/wordcloud/stop_words.py` (missing comma).
- No tests, CI, or deployment config tracked in repository.

