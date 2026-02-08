from __future__ import annotations

from app.ingestion.reddit_client import RedditClient
from app.ingestion.raw_ingest_service import upsert_raw_posts
from app.database.session import init_db


def run_pipeline_once(
    subreddits: list[str],
    fetch_limit: int,
    process_limit: int,
) -> dict:
    if not subreddits:
        return {
            "ingest": {"fetched": 0, "touched": 0},
            "process": {"claimed": 0, "processed": 0, "failed": 0, "skipped": 0},
        }
    if fetch_limit <= 0 or process_limit <= 0:
        raise ValueError("fetch_limit and process_limit must be > 0")

    init_db()
    rows = RedditClient().fetch_raw_rows(subreddits=subreddits, limit=fetch_limit)
    touched = upsert_raw_posts(rows)

    from app.processing.sentiment_processing_service import process_batch

    process_stats = process_batch(limit=process_limit)

    return {
        "ingest": {"fetched": len(rows), "touched": touched},
        "process": process_stats,
    }


def run_pipeline(subreddits: list[str], fetch_limit: int, process_limit: int) -> dict:
    return run_pipeline_once(subreddits, fetch_limit, process_limit)



