from datetime import timedelta
from celery import Celery

from app.orchestration.pipeline_service import run_pipeline_once
from app.settings import REDDIT_SUBREDDITS, REDDIT_FETCH_LIMIT, BATCH_PROCESS_LIMIT


app = Celery(
    "alphaone_reddit_worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

app.conf.beat_schedule = {
    "run_batch_periodically": {
        "task": "app.celery_app.run_batch",
        "schedule": timedelta(minutes=120),
    },
}
app.conf.timezone = "UTC"
app.conf.broker_connection_retry_on_startup = True


@app.task(name="app.celery_app.run_batch")
def run_batch():
    result = run_pipeline_once(REDDIT_SUBREDDITS, REDDIT_FETCH_LIMIT, BATCH_PROCESS_LIMIT)
    print(f"[run_batch] result={result}")
    return result

