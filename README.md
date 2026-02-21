<h1 align="center">AlphaOne</h1>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-active-success" />
  <img alt="Backend API" src="https://img.shields.io/badge/api-Spring%20Boot%203.5-brightgreen" />
  <img alt="Worker" src="https://img.shields.io/badge/worker-Celery%2BRedis-red" />
  <img alt="Frontend" src="https://img.shields.io/badge/frontend-React%20%2B%20Vite-blue" />
  <img alt="Database" src="https://img.shields.io/badge/database-PostgreSQL-336791" />
  <img alt="ML" src="https://img.shields.io/badge/ML-PyTorch%20%2B%20Transformers-orange" />
</p>
<p align="center">
  Real-time stock sentiment tracker powered by Reddit and a fine-tuned BERT model.
</p>

---

## What Is AlphaOne?

AlphaOne monitors Reddit communities (r/wallstreetbets, r/stocks, r/investing, etc.) and analyzes what people are saying about individual stocks. For each stock mention, it determines whether the sentiment is **bullish**, **bearish**, or **neutral** — and surfaces this through a web dashboard with charts, evidence feeds, and word clouds.

The core challenge: a sentence like *"AAPL is great but TSLA is doomed"* contains **two different sentiments** for two different stocks. Off-the-shelf sentiment models produce a single label for the entire sentence. AlphaOne solves this with a **custom fine-tuned BERT model** that classifies sentiment **per stock** within the same sentence.

## How It Works

```text
  Reddit API
      |
      v
  Python worker (Celery + Redis)
      |  fetches posts on a schedule
      v
  PostgreSQL (raw_reddit_posts)
      |  stored as-is for reprocessing
      v
  NLP Pipeline
      |  1. split post into sentences
      |  2. detect which stocks are mentioned (100+ tickers)
      |  3. classify sentiment per stock (fine-tuned BERT)
      |  4. aggregate word frequencies
      v
  PostgreSQL (sentiment_data, topics, word_frequency)
      |
      v
  Spring Boot API (/api/v1/...)
      |
      v
  React Dashboard
```

**Five services** run together via Docker Compose:

| Service | Role |
|---------|------|
| `reddit-worker` | Python Celery worker + beat scheduler — ingests Reddit posts and runs NLP |
| `api` | Java/Spring Boot — serves processed data to the frontend |
| `frontend` | React + Vite — dashboard UI with charts, evidence feeds, word cloud |
| `redis` | Message broker for Celery task queue |
| `postgresql` | External managed database (e.g., Neon) — not containerized |

---

## Sentiment Model

### The Problem

Standard sentiment models like [FinBERT](https://huggingface.co/ProsusAI/finbert) produce one label per sentence. They cannot distinguish that *"AAPL is great but TSLA is doomed"* is bullish for Apple and bearish for Tesla. This is a known NLP problem called **Aspect-Based Sentiment Analysis (ABSA)** — classifying sentiment toward a specific entity within a sentence.

### Our Approach

We fine-tuned a BERT-class model to take both a sentence and a target stock as input:

```
Input:    "AAPL is great but TSLA is doomed"  +  target: AAPL
Process:  "[TARGET] is great but [OTHER] is doomed"  (entity replacement)
Output:   bullish
```

The target stock is replaced with `[TARGET]` and all other stock tickers with `[OTHER]`, so the model learns to focus on the sentiment expressed toward the target entity regardless of which specific stock it is.

Training data: ~3,500 (sentence, stock, label) triples extracted from Reddit posts, labeled by a local LLM (qwen2.5:3b via Ollama) with a carefully balanced few-shot prompt.

### Results

We ran an ablation study across four transformer architectures. Best result: **80.5% accuracy** — up from **54.3%** using FinBERT out of the box with no fine-tuning.

| Model | Method | Accuracy | Macro F1 |
|-------|--------|----------|----------|
| FinBERT (no fine-tuning) | — | 54.3% | 0.433 |
| FinBERT | Fine-tuned | 75.9% | 0.703 |
| **RoBERTa (twitter-roberta)** | **Fine-tuned** | **79.6%** | **0.756** |
| **DeBERTa-v3-base** | **Fine-tuned** | **80.5%** | **0.757** |
| BERTweet | Fine-tuned | 75.1% | 0.700 |

Macro F1 measures balanced performance across all three classes (bullish/bearish/neutral). DeBERTa-v3 and RoBERTa tied on F1 — DeBERTa edges ahead on accuracy.

Full design details, hyperparameters, and per-run metrics: [`docs/FINBERT_FINETUNING_DESIGN.md`](docs/FINBERT_FINETUNING_DESIGN.md)

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **API** | Java 21, Spring Boot 3.5, Spring Data JPA, Spring Actuator |
| **Worker** | Python 3.11, Celery 5, Redis, SQLAlchemy, PRAW, spaCy, psycopg2 |
| **ML** | PyTorch, Hugging Face Transformers, PEFT (LoRA), scikit-learn |
| **Frontend** | React 18, React Router, Axios, Chart.js, react-d3-cloud, Vite |
| **Database** | PostgreSQL |
| **Deployment** | Docker + Docker Compose |

## Project Structure
```text
alphaone/
  api/                     # Java — Spring Boot read API
  backend/                 # Python — worker + ML training
    app/
      celery_app.py        # Celery configuration and beat schedule
      database/            # SQLAlchemy models and session
      ingestion/           # Reddit data ingestion (PRAW)
      processing/          # NLP pipeline: sentiment, topic tagging, word cloud
    ml/                    # ML training (not used at runtime)
      train_finbert.py     # Fine-tuning entry point
      eval_baseline.py     # Baseline evaluation (no fine-tuning)
      data/                # Training data: labeling, building, prompts
      models/              # Saved model checkpoints
  frontend/                # React dashboard application
  docs/                    # Design documents
  docker-compose.yml
```

---

## Getting Started

### Prerequisites

- Docker Desktop + Docker Compose plugin
- Reddit API credentials ([create an app](https://www.reddit.com/prefs/apps))
- PostgreSQL database (external — e.g., [Neon](https://neon.tech/), Supabase, or local)

### Configuration

**1. Root `.env`** — used by the Python worker:

```env
REDDIT_CLIENT_ID=<your_client_id>
REDDIT_SECRET_KEY=<your_secret>
REDDIT_USERNAME=<your_username>
REDDIT_PASSWORD=<your_password>
REDDIT_SUBREDDITS=wallstreetbets,stocks,investing
REDDIT_FETCH_LIMIT=100
BATCH_PROCESS_LIMIT=100
```

**2. `api/.env`** — used by the Spring Boot API:

```env
DATABASE_URL=jdbc:postgresql://<host>:<port>/<db>?sslmode=require
DATABASE_USERNAME=<db_user>
DATABASE_PASSWORD=<db_password>
```

**3. `frontend/.env.local`** (only for local Vite dev, not needed for Docker):

```env
VITE_API_URL=http://127.0.0.1:8080
```

### Run with Docker

```bash
docker compose up --build
```

Then open `http://localhost:5173` for the dashboard and `http://127.0.0.1:8080/actuator/health` to verify the API.

### Run Locally (Without Docker)

**API:**
```bash
cd api
./mvnw spring-boot:run
```

**Frontend:**
```bash
cd frontend
npm install && npm run dev
```

**Worker:**
```bash
cd backend
pip install -r requirements.txt
celery -A app.celery_app worker --beat --loglevel=info --concurrency=2
```

### ML Training (Optional)

Training requires a CUDA-capable GPU. The worker uses a pre-trained model checkpoint at runtime — you only need this if you want to retrain the sentiment model.

```bash
cd backend
pip install -r ml/requirements.txt

# Fine-tune with default settings (RoBERTa + LoRA)
python -m ml.train_finbert

# Full fine-tune with DeBERTa-v3 (best accuracy)
python -m ml.train_finbert --model microsoft/deberta-v3-base --no-lora --freeze-layers 0 --lr 2e-5 --epochs 10
```

---

## API Endpoints

Base URL: `http://127.0.0.1:8080`

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/assets/tracked` | List all tracked stock tickers |
| `GET /api/v1/signals/social-sentiment/{ticker}/evidence` | Sentence-level evidence for a ticker |
| `GET /api/v1/signals/social-sentiment/{ticker}/daily` | Daily sentiment aggregation |
| `GET /api/v1/signals/social-sentiment/summary/{topicSlug}` | Summary stats for a topic |
| `GET /api/v1/signals/social-sentiment/wordcloud` | Word frequency data for word cloud |
| `GET /actuator/health` | Service health check |

## Operations Notes

- The worker runs as a non-root user (`appuser`) inside Docker.
- CORS is configured for `localhost:5173`, `127.0.0.1:5173`, and `alphaone.run.place`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `url must start with jdbc` | The Spring Boot `DATABASE_URL` must use JDBC format: `jdbc:postgresql://...` |
| Worker can't connect to Redis | Make sure `redis` service is running: `docker compose ps` |
| No data appearing in dashboard | Check worker logs: `docker compose logs reddit-worker`. Posts take ~1 minute to appear after first startup. |
