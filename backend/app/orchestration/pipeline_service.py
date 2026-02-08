from __future__ import annotations

import logging

from app.ingestion.reddit_client import RedditClient
from app.ingestion.raw_ingest_service import upsert_raw_posts
from app.database.session import init_db

logger = logging.getLogger(__name__)


def run_pipeline_once(
    subreddits: list[str],
    fetch_limit: int,
    process_limit: int,
) -> dict:
    logger.info(
        "pipeline_run_started subreddits=%s fetch_limit=%s process_limit=%s",
        len(subreddits),
        fetch_limit,
        process_limit,
    )
    if not subreddits:
        logger.warning("pipeline_run_no_subreddits")
        return {
            "ingest": {"fetched": 0, "touched": 0},
            "process": {"claimed": 0, "processed": 0, "failed": 0, "skipped": 0},
        }
    if fetch_limit <= 0 or process_limit <= 0:
        logger.error(
            "pipeline_run_invalid_limits fetch_limit=%s process_limit=%s",
            fetch_limit,
            process_limit,
        )
        raise ValueError("fetch_limit and process_limit must be > 0")

    try:
        init_db()
        rows = RedditClient().fetch_raw_rows(subreddits=subreddits, limit=fetch_limit)
        touched = upsert_raw_posts(rows)

        from app.processing.sentiment_processing_service import process_batch

        process_stats = process_batch(limit=process_limit)
        result = {
            "ingest": {"fetched": len(rows), "touched": touched},
            "process": process_stats,
        }
        logger.info(
            "pipeline_run_completed fetched=%s touched=%s claimed=%s processed=%s failed=%s skipped=%s",
            result["ingest"]["fetched"],
            result["ingest"]["touched"],
            result["process"]["claimed"],
            result["process"]["processed"],
            result["process"]["failed"],
            result["process"]["skipped"],
        )
        return result
    except Exception:
        logger.exception("pipeline_run_failed")
        raise


def run_pipeline(subreddits: list[str], fetch_limit: int, process_limit: int) -> dict:
    return run_pipeline_once(subreddits, fetch_limit, process_limit)



