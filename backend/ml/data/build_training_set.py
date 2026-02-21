"""
Build the 3NF training set from raw_reddit_posts.

Pipeline:
  1. Scan raw posts sequentially by ID (cursor-based)
  2. Split each post into sentences (spaCy)
  3. For each sentence, run normalize_and_tag_sentence()
  4. Keep only sentences with at least one detected subject
  5. Write to training_sentences (text stored once)
  6. Write subjects to training_sentence_subjects (one row per subject)

Posts with no matching topics are simply skipped — no placeholders, no status changes.
Resumable: without --reset, continues from where it left off.

Usage:
    cd backend
    python -m ml.data.build_training_set
    python -m ml.data.build_training_set --batch-size 200
    python -m ml.data.build_training_set --reset   # truncate training tables and rebuild
"""

import logging
import argparse

from sqlalchemy import func as sql_func, text as sql_text
from sqlalchemy.exc import IntegrityError

from app.database.session import init_db, SessionLocal
from app.database.models import RawRedditPost, TrainingSentence, TrainingSentenceSubject
from app.processing.text_utils import split_into_sentences
from app.processing.sentiment_tagger.tagger_logic import normalize_and_tag_sentence, _GENERAL_TOPICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def build_training_set(batch_size: int = BATCH_SIZE) -> dict:
    """
    Process raw posts into the 3NF training tables.
    Scans sequentially by raw_post_id — no LEFT JOIN, no placeholders.
    Uses savepoints so a single dupe doesn't roll back the entire batch.
    """
    init_db()
    db = SessionLocal()

    total_posts = 0
    total_sentences = 0
    total_subjects = 0
    skipped_dupes = 0

    # Resume from where we left off (max raw_post_id already in training_sentences)
    last_id = db.query(
        sql_func.coalesce(sql_func.max(TrainingSentence.raw_post_id), 0)
    ).scalar()
    logger.info("build_training_set_started last_id=%d batch_size=%d", last_id, batch_size)

    try:
        while True:
            posts = (
                db.query(RawRedditPost)
                .filter(RawRedditPost.id > last_id)
                .order_by(RawRedditPost.id.asc())
                .limit(batch_size)
                .all()
            )
            if not posts:
                break

            batch_sentences = 0
            batch_subjects = 0

            for post in posts:
                try:
                    if not post.text or not isinstance(post.text, str):
                        continue

                    sentences = split_into_sentences(post.text)

                    for i, sentence in enumerate(sentences):
                        normalized_text, topics = normalize_and_tag_sentence(sentence)
                        # Drop general topics (MACRO, TECHNOLOGY) — too broad for training
                        topics = topics - _GENERAL_TOPICS
                        if not topics:
                            continue

                        train_sent = TrainingSentence(
                            raw_post_id=post.id,
                            sentence_index=i,
                            normalized_text=normalized_text,
                            subreddit=post.subreddit,
                            created_utc=post.created_utc,
                        )

                        nested = db.begin_nested()
                        try:
                            db.add(train_sent)
                            db.flush()

                            for subject in topics:
                                db.add(TrainingSentenceSubject(
                                    sentence_id=train_sent.id,
                                    subject=subject,
                                ))
                            nested.commit()

                            batch_sentences += 1
                            batch_subjects += len(topics)
                        except IntegrityError:
                            nested.rollback()
                            skipped_dupes += 1
                except Exception:
                    logger.exception("Error processing post id=%d, skipping", post.id)
                    continue

            # Advance cursor — always moves forward regardless of topic matches
            last_id = posts[-1].id
            db.commit()

            total_posts += len(posts)
            total_sentences += batch_sentences
            total_subjects += batch_subjects

            logger.info(
                "batch_done posts=%d sentences=%d subjects=%d last_id=%d | cumulative posts=%d sentences=%d subjects=%d",
                len(posts), batch_sentences, batch_subjects, last_id,
                total_posts, total_sentences, total_subjects,
            )

    except Exception:
        db.rollback()
        logger.exception("build_training_set_failed at last_id=%d", last_id)
        raise
    finally:
        db.close()

    result = {
        "posts_processed": total_posts,
        "sentences_created": total_sentences,
        "subjects_created": total_subjects,
        "skipped_dupes": skipped_dupes,
    }
    logger.info("build_training_set_completed %s", result)
    return result


def reset_training_tables():
    """Truncate training tables only. Does not touch raw_reddit_posts."""
    db = SessionLocal()
    try:
        db.execute(sql_text(
            "TRUNCATE training_sentence_subjects, training_sentences RESTART IDENTITY CASCADE"
        ))
        db.commit()
        logger.info("reset_complete: training tables truncated")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Build 3NF training set from raw posts")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Posts per batch (default: 100)")
    parser.add_argument("--reset", action="store_true", help="Truncate training tables before rebuilding")
    args = parser.parse_args()

    init_db()

    if args.reset:
        reset_training_tables()

    print("=== Build Training Set (3NF) ===")
    print(f"Batch size: {args.batch_size}")
    print()

    result = build_training_set(batch_size=args.batch_size)

    print()
    print("=== Results ===")
    print(f"Posts processed:  {result['posts_processed']}")
    print(f"Sentences:        {result['sentences_created']}")
    print(f"Subject entries:  {result['subjects_created']}")
    print(f"Skipped (dupes):  {result['skipped_dupes']}")


if __name__ == "__main__":
    main()
