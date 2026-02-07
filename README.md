# alphaone

alphaone is a full-stack social sentiment analytics system for market topics and tickers.

It ingests Reddit data, processes sentiment at sentence level, stores results in PostgreSQL, and serves analytics through a Spring Boot API to a React frontend.

## Features

1. End-to-end pipeline from social text to dashboard-ready metrics.
2. Sentence-level sentiment scoring with topic tagging.
3. Topic and ticker analytics (`AAPL`, `NVDA`, `MACRO`, `TECHNOLOGY`, etc.).
4. Evidence view (latest sentiment sentences for a ticker).
5. Daily sentiment time-series for charting.
6. Topic summary sentiment score.
7. Word cloud endpoint (`text`, `value`) from stored frequency data.
8. CORS-enabled Spring API for local and deployed frontend integration.
9. Separated service structure for system design growth:
   - Python data pipeline (`backend/`)
   - Java API service (`api/`)
   - React frontend (`frontend/`)

## Current Architecture

1. `backend/` (Python)
   - Reddit collection and NLP processing.
   - Writes processed rows into PostgreSQL.
2. `api/` (Java Spring Boot)
   - Read-focused API over PostgreSQL.
   - Route parity with prior FastAPI endpoints.
3. `frontend/` (React + Vite)
   - Fetches sentiment endpoints and renders summary UI.

Data flow:

`Reddit -> Python processing -> PostgreSQL -> Spring Boot API -> React frontend`

## Implemented API Endpoints (Spring)

Base URL: `http://127.0.0.1:8080`

1. `GET /api/v1/assets/tracked`
2. `GET /api/v1/signals/social-sentiment/{ticker}/evidence`
3. `GET /api/v1/signals/social-sentiment/{ticker}/daily`
4. `GET /api/v1/signals/social-sentiment/summary/{topicSlug}`
5. `GET /api/v1/signals/social-sentiment/wordcloud`

Operational endpoint:

1. `GET /actuator/health`

## Tech Stack

1. Backend data pipeline: Python, PRAW, spaCy, FinBERT, SQLAlchemy.
2. API: Java 21, Spring Boot 3.5, Spring Data JPA, PostgreSQL.
3. Frontend: React 18, Vite, Axios, Chart.js.
4. Database: PostgreSQL (Neon-compatible connection setup).

## Project Structure

```text
alphaone/
  backend/     # Python ingestion + NLP + batch processing
  api/         # Spring Boot API service
  frontend/    # React frontend
  docs/        # Plans and technical documentation
```

## Local Setup

## Prerequisites

1. Python 3.10+ (for `backend/` workflow).
2. Node.js 18+ and npm (for `frontend/`).
3. Java 21 and Maven wrapper (for `api/`).
4. PostgreSQL database connection string and credentials.

## 1) Configure environment files

1. Root `.env` is used by Python pipeline (`backend/`).
2. `api/.env` is used by Spring API.
3. `frontend/.env.local` can override API base URL.

Example `frontend/.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8080
```

## 2) Run Spring API

```bash
cd api
./mvnw spring-boot:run
```

Check health:

```bash
curl http://127.0.0.1:8080/actuator/health
```

## 3) Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default dev URL is usually:

`http://localhost:5173`

## 4) (Optional) Run Python pipeline

Use `backend/` scripts to fetch/process new data before querying Spring endpoints.

Typical files:

1. `backend/core/reddit_client.py`
2. `backend/batch_processor.py`

## Notes on Data and Semantics

1. `summary/{topicSlug}` currently computes average over all available rows for that topic.
2. `/{ticker}/daily` returns a list of daily averages.
3. `/{ticker}/evidence` returns latest 5 linked sentiment records.
4. `wordcloud` returns top words by stored frequency.

## Security Notes

1. Do not commit real `.env` secrets.
2. Rotate credentials immediately if exposed.
3. Use `.env.example` templates for onboarding.

## Documentation

1. Future implementation roadmap: `docs/FUTURE_PLAN.md`
2. Spring migration and review notes: `docs/SPRING_BOOT_KNOWLEDGE_REVIEW.md`

## Resume-Oriented System Design Direction

1. Keep API, processing, and ingestion as explicit service boundaries.
2. Add observability (`metrics`, tracing, structured logs).
3. Add queue-based processing and idempotent pipelines.
4. Add integration tests and CI/CD for production-ready story.

## License

No license file is currently defined.
