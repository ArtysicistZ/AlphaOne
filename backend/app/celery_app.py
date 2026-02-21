import json
import logging
import time
import uuid

from datetime import timedelta
from celery import Celery

from app.orchestration.pipeline_service import run_pipeline_once
from app.settings import REDDIT_SUBREDDITS, REDDIT_FETCH_LIMIT, BATCH_PROCESS_LIMIT

logger = logging.getLogger(__name__)

app = Celery(
    "alphaone_reddit_worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
)

app.conf.beat_schedule = {
    "run_batch_periodically": {
        "task": "app.celery_app.run_batch",
        "schedule": timedelta(minutes=120),  # Run every 2 hours
    },
}
app.conf.timezone = "UTC"
app.conf.broker_connection_retry_on_startup = True


@app.task(
    bind=True,
    name="app.celery_app.run_batch",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def run_batch(self):
    run_id = str(uuid.uuid4())
    started = time.time()
    logger.info(json.dumps({
        "event": "run_batch_started",
        "run_id": run_id,
        "timestamp": started,
    }))

    try:
        result = run_pipeline_once(REDDIT_SUBREDDITS, REDDIT_FETCH_LIMIT)

        duration_ms = int((time.time() - started) * 1000)
        logger.info(json.dumps({
            "event": "run_batch_completed",
            "run_id": run_id,
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "result": result,
        }))
        return result
    
    except Exception as e:
        duration_ms = int((time.time() - started) * 1000)
        logger.error(json.dumps({
            "event": "run_batch_failed",
            "run_id": run_id,
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "error": str(e),
            "attempt": self.request.retries + 1,
            "max_retries": self.max_retries,
            "task_id": self.request.id,
        }))
        raise
