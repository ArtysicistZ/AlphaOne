<h1 align="center">AlphaOne</h1>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-active-success" />
  <img alt="Backend API" src="https://img.shields.io/badge/api-Spring%20Boot%203.5-brightgreen" />
  <img alt="Worker" src="https://img.shields.io/badge/worker-Celery%2BRedis-red" />
  <img alt="Frontend" src="https://img.shields.io/badge/frontend-React%20%2B%20Vite-blue" />
  <img alt="Database" src="https://img.shields.io/badge/database-PostgreSQL-336791" />
</p>
<p align="center">
  Full-stack social sentiment intelligence platform for market topics and tickers.
</p>
<p align="center">
  AlphaOne ingests Reddit content, runs NLP sentiment/topic extraction, stores normalized results in PostgreSQL, and serves analytics through a Spring Boot API to a React dashboard.
</p>

## Features
- End-to-end data flow: ingest -> process -> persist -> serve -> visualize.
- Periodic asynchronous processing with Celery + Redis beat/worker.
- Idempotent raw ingestion using `source_id` upsert semantics.
- Sentence-level sentiment scoring (FinBERT) with topic tagging.
- Read API for tracked assets, evidence feeds, daily sentiment, topic summary, and word cloud.
- Dockerized multi-service runtime (`api`, `frontend`, `reddit-worker`, `redis`).
- Health checks via Spring Actuator.

## How It Works
```text
Reddit API
  -> Python ingestion client (fetch raw rows)
  -> raw_reddit_posts upsert (idempotent by source_id)
  -> claim "new" rows (FOR UPDATE SKIP LOCKED)
  -> NLP processing (sentence split, sentiment, topic tagging, word counts)
  -> PostgreSQL tables (sentiment_data, topics, word_frequency, associations)
  -> Spring Boot read API (/api/v1/...)
  -> React dashboard (charts, evidence, word cloud)
```

## Architecture

Services:
- `frontend`: React/Vite static bundle served by Nginx.
- `api`: Spring Boot service exposing analytics endpoints.
- `reddit-worker`: Celery worker + beat scheduler for ingestion/processing.
- `redis`: Broker/result backend for Celery.
- `postgresql`: external managed DB (for example Neon), not containerized in this repo.

Core data model:
- `raw_reddit_posts`: raw source payloads and ingestion status.
- `sentiment_data`: processed sentence-level sentiment events.
- `topics`: tracked topic taxonomy (`slug`, `name`).
- `sentiment_topic_association`: many-to-many mapping.
- `word_frequency`: daily aggregated word counts for word cloud.

## API Endpoints

Base URL: `http://127.0.0.1:8080`

- `GET /api/v1/assets/tracked`
- `GET /api/v1/signals/social-sentiment/{ticker}/evidence`
- `GET /api/v1/signals/social-sentiment/{ticker}/daily`
- `GET /api/v1/signals/social-sentiment/summary/{topicSlug}`
- `GET /api/v1/signals/social-sentiment/wordcloud`
- `GET /actuator/health`

## Tech Stack
- API: Java 21, Spring Boot 3.5, Spring Data JPA, Spring Actuator.
- Worker: Python 3.11, Celery 5, Redis, SQLAlchemy, PRAW, spaCy, Transformers, PyTorch (CPU), psycopg2.
- Frontend: React 18, React Router, Axios, Chart.js, react-d3-cloud, Vite.
- Database: PostgreSQL.
- Deployment: Docker + Docker Compose.

## Project Structure
```text
alphaone/
  api/                 # Spring Boot read API
  backend/             # Python worker pipeline
    app/
      celery_app.py
      settings.py
      database/
      ingestion/
      orchestration/
      processing/
  frontend/            # React application
  docs/                # Design notes and plans
  docker-compose.yml
```

## Requirements
- Docker Desktop + Docker Compose plugin.
Optional local runtimes:
- Java 21 (for `api` local dev outside Docker).
- Node.js 20+ (for `frontend` local dev outside Docker).
- Python 3.11 (for worker local dev outside Docker).
- External PostgreSQL database credentials.
- Reddit API credentials.

## Configuration

### 1) Root `.env` (used by worker service)
Create `.env` at project root with at least:

```env
REDDIT_CLIENT_ID=
REDDIT_SECRET_KEY=
REDDIT_USERNAME=
REDDIT_PASSWORD=
REDDIT_SUBREDDITS=wallstreetbets,stocks,investing
REDDIT_FETCH_LIMIT=100
BATCH_PROCESS_LIMIT=100
```

### 2) `api/.env` (used by Spring Boot)
`application.yml` reads these keys:

```env
DATABASE_URL=jdbc:postgresql://<host>:<port>/<db>?sslmode=require
DATABASE_USERNAME=<db_user>
DATABASE_PASSWORD=<db_password>
```

### 3) Frontend API URL
For local Vite development, create `frontend/.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8080
```

For Docker build, Compose passes `VITE_API_URL` as build arg.

## Quick Start (Docker)
1. Build and run:
```bash
docker compose up --build
```
2. Check running services:
```bash
docker compose ps
```
3. Verify API health:
```bash
curl http://127.0.0.1:8080/actuator/health
```
4. Open UI:
`http://localhost:5173`

## Local Development (Without Docker)

### API
```bash
cd api
./mvnw spring-boot:run
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Worker
```bash
cd backend
pip install -r requirements.txt
celery -A app.celery_app worker --beat --loglevel=info --concurrency=2
```

## Operations Notes
- Celery beat schedule is currently configured at `1 minute` in `backend/app/celery_app.py`.
- Worker runs as non-root user in Docker (`appuser`).
- Spring API healthcheck is configured in `docker-compose.yml`.
CORS allows:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `https://alphaone.run.place`

## Known Gaps
- End-to-end automated integration tests are still pending.

## Troubleshooting
- `url must start with jdbc`: Spring `DATABASE_URL` must be JDBC format.
