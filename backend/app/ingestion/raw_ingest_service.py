from datetime import datetime, timezone
import logging

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import case, func, text

from app.database.session import SessionLocal
from app.database.models import RawRedditPost

logger = logging.getLogger(__name__)


def upsert_raw_posts(rows: list[dict]) -> int:
    db = SessionLocal()
    touched = 0
    logger.info("raw_upsert_started rows=%s", len(rows))
    try:
        for row in rows:
            stmt = insert(RawRedditPost).values(
                source_id=row["source_id"],
                source_type=row["source_type"],
                text=row["text"],
                subreddit=row.get("subreddit"),
                created_utc=row.get("created_utc"),
                edited_utc=row.get("edited_utc"),
                status="new",
                fetched_at=datetime.now(timezone.utc),
                last_seen_at=datetime.now(timezone.utc),
                fetch_count=1,
                content_version=1,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[RawRedditPost.source_id],
                set_={
                    "last_seen_at": func.now(),
                    "fetch_count": RawRedditPost.fetch_count + 1,
                    "edited_utc": func.coalesce(row.get("edited_utc"), RawRedditPost.edited_utc),
                    "text": row["text"],
                    "content_version": case(
                        (
                            RawRedditPost.text.is_distinct_from(row["text"]),
                            RawRedditPost.content_version + 1,
                        ),
                        else_=RawRedditPost.content_version,
                    ),
                    "status": case(
                        (
                            RawRedditPost.text.is_distinct_from(row["text"]),
                            "new",
                        ),
                        else_=RawRedditPost.status,
                    ),
                },
            )
            db.execute(stmt)
            touched += 1
        db.commit()
        logger.info("raw_upsert_completed touched=%s", touched)
        return touched
    except Exception:
        db.rollback()
        logger.exception("raw_upsert_failed touched=%s", touched)
        raise
    finally:
        db.close()


def claim_new_raw_posts(limit: int = 100) -> list[dict]:
    db = SessionLocal()
    logger.debug("raw_claim_started limit=%s", limit)
    try:
        sql = text(
            """
            WITH picked AS (
                SELECT id
                FROM raw_reddit_posts
                WHERE status = 'new'
                ORDER BY fetched_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT :limit
            )
            UPDATE raw_reddit_posts r
            SET status = 'processing',
                last_seen_at = NOW()
            FROM picked
            WHERE r.id = picked.id
            RETURNING
                r.id, r.source_id, r.source_type, r.text, r.subreddit, r.created_utc, r.content_version;
            """
        )
        rows = db.execute(sql, {"limit": limit}).mappings().all()
        db.commit()
        claimed = [dict(r) for r in rows]
        logger.info("raw_claim_completed limit=%s claimed=%s", limit, len(claimed))
        return claimed
    except Exception:
        db.rollback()
        logger.exception("raw_claim_failed limit=%s", limit)
        raise
    finally:
        db.close()


def mark_processed(raw_id: int) -> None:
    db = SessionLocal()
    try:
        updated = (
            db.query(RawRedditPost)
            .filter(RawRedditPost.id == raw_id)
            .update({"status": "processed"})
        )
        db.commit()
        logger.debug("raw_mark_processed raw_id=%s updated=%s", raw_id, updated)
    except Exception:
        db.rollback()
        logger.exception("raw_mark_processed_failed raw_id=%s", raw_id)
        raise
    finally:
        db.close()


def mark_failed(raw_id: int) -> None:
    db = SessionLocal()
    try:
        updated = db.query(RawRedditPost).filter(RawRedditPost.id == raw_id).update({"status": "failed"})
        db.commit()
        logger.debug("raw_mark_failed raw_id=%s updated=%s", raw_id, updated)
    except Exception:
        db.rollback()
        logger.exception("raw_mark_failed_failed raw_id=%s", raw_id)
        raise
    finally:
        db.close()
