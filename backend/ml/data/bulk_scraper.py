"""
Bulk Reddit scraper for ML training data collection.

Scrapes multiple listing endpoints (hot, new, top, rising, comments) across
finance-related subreddits to build a large raw post dataset in raw_reddit_posts.

Reuses existing RedditClient (credentials, normalization) and upsert_raw_posts()
for dedup-aware storage.

Usage:
    cd backend
    python -m ml.data.bulk_scraper
    python -m ml.data.bulk_scraper --target 8000
    python -m ml.data.bulk_scraper --subreddits wallstreetbets stocks investing
"""

import logging
import argparse

from app.ingestion.reddit_client import RedditClient
from app.ingestion.raw_ingest_service import upsert_raw_posts
from app.database.session import init_db, SessionLocal
from app.database.models import RawRedditPost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Scrape configuration ──────────────────────────────────────────────────

# Balanced subreddit list for sentiment training data.
# Removed neutral-heavy subs (SecurityAnalysis, finance, economy) that
# flood the dataset with factual/news content and rarely mention tickers.
# Added bullish-leaning subs (dividends, ValueInvesting) where users
# discuss stocks they're accumulating and post bullish theses.
BULK_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "stockmarket",
    "options",
    "dividends",
    "ValueInvesting",
]

# Each tuple: (listing_method, extra_kwargs, limit)
# These are PRAW Subreddit listing methods for submissions.
SUBMISSION_ENDPOINTS = [
    ("hot",    {},                        300),
    ("new",    {},                        300),
    ("top",    {"time_filter": "week"},   250),
    ("top",    {"time_filter": "month"},  250),
    ("top",    {"time_filter": "year"},   250),
    ("rising", {},                        100),
]

COMMENT_LIMIT = 300
UPSERT_BATCH_SIZE = 500

# ── Scope estimate ────────────────────────────────────────────────────────
#
# Per subreddit:
#   submissions: 300 + 300 + 250 + 250 + 250 + 100 = 1,450
#   comments:    300
#   total:       1,750
#
# 7 subreddits × 1,750 = 12,250 raw items before dedup.
# After dedup (overlap between hot/top, cross-posts): ~60-70% unique
# Expected unique: ~8,400 – 9,800
# Target 6,000 ➜ comfortable margin.
# ──────────────────────────────────────────────────────────────────────────


def get_raw_post_count() -> int:
    db = SessionLocal()
    try:
        return db.query(RawRedditPost).count()
    finally:
        db.close()


def flush_buffer(buffer: list[dict]) -> int:
    """Upsert buffered rows and return touched count."""
    if not buffer:
        return 0
    touched = upsert_raw_posts(buffer)
    logger.info("batch_upserted touched=%d buffer_size=%d", touched, len(buffer))
    return touched


def scrape_subreddit(reddit, sub_name: str) -> list[dict]:
    """Scrape all endpoints for a single subreddit. Returns normalized rows."""
    subreddit = reddit.subreddit(sub_name)
    rows = []

    # Submissions from multiple listing endpoints
    for method_name, kwargs, limit in SUBMISSION_ENDPOINTS:
        endpoint_label = method_name if not kwargs else f"{method_name}({kwargs.get('time_filter', '')})"
        try:
            listing = getattr(subreddit, method_name)(limit=limit, **kwargs)
            count = 0
            for submission in listing:
                row = RedditClient._normalize_submission(submission, sub_name)
                if row:
                    rows.append(row)
                    count += 1
            logger.info(
                "endpoint_done sub=%s endpoint=%s fetched=%d",
                sub_name, endpoint_label, count,
            )
        except Exception:
            logger.exception("endpoint_failed sub=%s endpoint=%s", sub_name, endpoint_label)

    # Comments
    try:
        count = 0
        for comment in subreddit.comments(limit=COMMENT_LIMIT):
            row = RedditClient._normalize_comment(comment, sub_name)
            if row:
                rows.append(row)
                count += 1
        logger.info("endpoint_done sub=%s endpoint=comments fetched=%d", sub_name, count)
    except Exception:
        logger.exception("endpoint_failed sub=%s endpoint=comments", sub_name)

    return rows


def bulk_scrape(subreddits: list[str], target: int = 6000) -> dict:
    """
    Main entry point. Scrapes all subreddits and upserts into raw_reddit_posts.
    Stops early if target count is reached.
    """
    init_db()

    start_count = get_raw_post_count()
    logger.info(
        "bulk_scrape_started target=%d current_count=%d subreddits=%s",
        target, start_count, subreddits,
    )

    client = RedditClient()
    reddit = client._get_client()

    total_fetched = 0
    total_upserted = 0
    buffer: list[dict] = []

    for sub_name in subreddits:
        # Check if we already hit target
        current_count = get_raw_post_count()
        if current_count >= target:
            logger.info("target_reached count=%d target=%d", current_count, target)
            break

        logger.info(
            "subreddit_started sub=%s current_db_count=%d",
            sub_name, current_count,
        )

        rows = scrape_subreddit(reddit, sub_name)
        total_fetched += len(rows)
        buffer.extend(rows)

        # Flush buffer if large enough
        if len(buffer) >= UPSERT_BATCH_SIZE:
            total_upserted += flush_buffer(buffer)
            buffer = []

        logger.info(
            "subreddit_completed sub=%s fetched=%d total_fetched=%d",
            sub_name, len(rows), total_fetched,
        )

    # Flush remaining
    total_upserted += flush_buffer(buffer)

    final_count = get_raw_post_count()
    new_rows = final_count - start_count

    logger.info(
        "bulk_scrape_completed fetched=%d upserted=%d new_rows=%d final_count=%d target=%d",
        total_fetched, total_upserted, new_rows, final_count, target,
    )

    return {
        "fetched": total_fetched,
        "upserted": total_upserted,
        "new_rows": new_rows,
        "final_count": final_count,
        "target": target,
    }


def main():
    parser = argparse.ArgumentParser(description="Bulk scrape Reddit for ML training data")
    parser.add_argument("--target", type=int, default=6000, help="Target raw post count (default: 6000)")
    parser.add_argument("--subreddits", nargs="+", default=None, help="Override subreddit list")
    args = parser.parse_args()

    subreddits = args.subreddits or BULK_SUBREDDITS

    print("=== Bulk Reddit Scraper ===")
    print(f"Target:     {args.target} raw posts")
    print(f"Subreddits: {', '.join(subreddits)}")
    print(f"Endpoints:  {len(SUBMISSION_ENDPOINTS)} submission types + comments")
    print()

    result = bulk_scrape(subreddits, target=args.target)

    print()
    print("=== Results ===")
    print(f"Fetched:     {result['fetched']}")
    print(f"Upserted:    {result['upserted']}")
    print(f"New rows:    {result['new_rows']}")
    print(f"Final count: {result['final_count']}")
    print(f"Target:      {result['target']}")

    if result["final_count"] >= result["target"]:
        print(f"\nTarget reached!")
    else:
        deficit = result["target"] - result["final_count"]
        print(f"\n{deficit} rows short of target. Run again or add more subreddits.")


if __name__ == "__main__":
    main()
