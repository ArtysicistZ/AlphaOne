# alphaone Future Plan (High-ROI System Design Focus)

This plan tracks resume-impactful system design work, ordered by ROI (signal-to-effort). It now includes implementation status.

## Progress Update (2026-02-07)

### Completed: Spring Boot API integration (backend API layer rewrite)

1. Created Java API service in `api/` with Spring Boot.
2. Configured PostgreSQL connection from `.env` for Spring.
3. Added JPA entities for existing tables:
   - `topics`
   - `sentiment_data`
   - `word_frequency`
   - join mapping via `sentiment_topic_association`
4. Added Spring Data repositories:
   - `TopicRepository`
   - `SentimentDataRepository`
   - `WordFrequencyRepository`
5. Added DTO layer for API contracts.
6. Implemented service layer:
   - topic listing
   - evidence retrieval
   - summary aggregation
   - daily time-series aggregation
   - word cloud retrieval
7. Implemented controllers with route parity for current FastAPI endpoints:
   - `GET /api/v1/assets/tracked`
   - `GET /api/v1/signals/social-sentiment/{ticker}/evidence`
   - `GET /api/v1/signals/social-sentiment/summary/{topicSlug}`
   - `GET /api/v1/signals/social-sentiment/{ticker}/daily`
   - `GET /api/v1/signals/social-sentiment/wordcloud`
8. Added CORS configuration for frontend origins.
9. Verified end-to-end API flow: DB -> Spring service -> JSON response.

### Remaining to fully switch runtime to Spring API

1. Point frontend runtime to Spring (`VITE_API_URL=http://127.0.0.1:8080`).
2. Run full frontend smoke test against Spring endpoints.
3. Standardize naming typo in DTO (`WorldCloudItemDto` -> `WordCloudItemDto`) after confirming no references break.
4. Add exception response standardization (`@ControllerAdvice`) for consistent JSON error shape.

## Next Implementation Phases

## Phase 0: Baseline Hygiene (1-2 days)

1. Rotate credentials and remove secrets from VCS history.
2. Add `.env.example` for root and `api/`.
3. Ensure `backend/worker.py` is marked deprecated if unused.

## Phase 1: System Design Core (Highest ROI) (1-2 weeks)

1. Keep service boundaries explicit:
   - `backend/`: ingestion + NLP processing
   - `api/`: read API service
   - `frontend/`: presentation layer
2. Introduce event transport between ingestion and processing:
   - Redis Streams or Kafka/Redpanda
   - event schema: `raw_post`, `processed_sentence`, `daily_aggregate`
3. Make processing idempotent with deterministic keys and upserts.
4. Add Redis caching for read-heavy endpoints (word cloud, summaries).

## Phase 2: Observability + Reliability (High ROI) (1 week)

1. Structured JSON logs with request IDs.
2. Metrics (`/actuator/metrics`, queue lag, API latency).
3. Health endpoints and readiness checks.
4. OpenTelemetry tracing across ingestion -> processing -> API.

## Phase 3: Data Model + Storage (Medium-High ROI) (1 week)

1. Separate raw and processed storage paths.
2. Add aggregate table (`daily_topic_metrics`) for query speed.
3. Add backfill command for historical reprocessing windows.

## Phase 4: Frontend Design + UX (High ROI for Web Design) (1-2 weeks)

1. Replace inline styles with a consistent design system.
2. Add advanced visualizations and filtering UX.
3. Build an architecture narrative page for project storytelling.

## Phase 5: Testing + CI/CD (Medium ROI) (1 week)

1. Add API unit/integration tests (Spring + DB).
2. Add frontend component and API-contract tests.
3. Add GitHub Actions (lint, test, build).
4. Add Docker Compose for local multi-service run.

## Phase 6: Resume Packaging (High ROI) (2-3 days)

1. Update root `README.md` with architecture and data flow diagrams.
2. Add tradeoff notes (latency, consistency, cost).
3. Publish benchmark snapshots (latency, throughput, pipeline lag).
