import logging

from sqlalchemy.exc import IntegrityError

from app.processing.nlp_processor import get_sentiment_batch
from app.processing.text_utils import split_into_sentences
from app.processing.sentiment_tagger.tagger_logic import normalize_and_tag_sentence, MAX_TICKERS_PER_SENTENCE
from app.processing.entity_replacement import apply_entity_replacement, ALL_STOCK_TICKERS
from app.ingestion.raw_ingest_service import claim_new_raw_posts, mark_processed, mark_failed
from app.database.session import SessionLocal, init_db
from app.database.models import SentimentData, Topic

logger = logging.getLogger(__name__)


def process_all():
    logger.info("process_all_started")
    claimed_total = 0
    processed_total = 0
    failed_total = 0
    skipped_total = 0
    while True:
        batch_result = process_batch(limit=40)
        if batch_result["claimed"] == 0:
            break
        claimed_total += batch_result["claimed"]
        processed_total += batch_result["processed"]
        failed_total += batch_result["failed"]
        skipped_total += batch_result["skipped"]
    logger.info("process_all_completed")
    return {
        "status": "completed",
        "claimed": claimed_total,
        "processed": processed_total,
        "failed": failed_total,
        "skipped": skipped_total,
    }


def process_batch(limit: int = 40):
    init_db()
    logger.info("process_batch_started limit=%s", limit)
    claimed_rows = claim_new_raw_posts(limit=limit)
    if not claimed_rows:
        logger.info("No new raw posts to process. Exiting batch.")
        return {"claimed": 0, "processed": 0, "failed": 0, "skipped": 0}

    # ── Pass 1: collect work items (CPU-only: split, normalize, entity replace) ──

    work_items = []  # list of (item, sentence_idx, sentence, subject, inference_text)
    failed_count = 0

    for item in claimed_rows:
        full_text = item["text"]
        if not full_text or not isinstance(full_text, str):
            mark_failed(item["id"])
            failed_count += 1
            continue

        sentences = split_into_sentences(full_text)

        for i, sentence in enumerate(sentences):
            normalized_text, topics_in_sentence = normalize_and_tag_sentence(sentence)
            if not topics_in_sentence:
                continue
            if len(topics_in_sentence) > MAX_TICKERS_PER_SENTENCE:
                continue

            for subject in topics_in_sentence:
                if subject not in ALL_STOCK_TICKERS:
                    continue
                inference_text = apply_entity_replacement(normalized_text, subject)
                work_items.append((item, i, sentence, subject, inference_text))

    if not work_items:
        for item in claimed_rows:
            mark_processed(item["id"])
        return {"claimed": len(claimed_rows), "processed": 0, "failed": failed_count, "skipped": 0}

    # ── Pass 2: batch inference (single forward pass) ────────────────────────

    all_texts = [wi[4] for wi in work_items]
    all_results = get_sentiment_batch(all_texts)

    # ── Pass 3: write results to DB (batch commit) ─────────────────────────

    db = SessionLocal()
    processed_count = 0
    skipped_count = 0

    try:
        topic_cache = {}
        processed_item_ids = set()

        for (item, sent_idx, sentence, subject, _), (score, label) in zip(work_items, all_results):
            new_data_point = SentimentData(
                source_id=f"{item['source_id']}_v{item['content_version']}_s{sent_idx}_{subject}",
                source_type="topic_sentence",
                sentiment_label=label,
                sentiment_score=score,
                relevant_text=sentence,
            )
            db.add(new_data_point)

            if subject not in topic_cache:
                topic = db.query(Topic).filter(Topic.slug == subject).first()
                if not topic:
                    topic = Topic(slug=subject, name=subject)
                    db.add(topic)
                topic_cache[subject] = topic
            new_data_point.topics.append(topic_cache[subject])

            processed_item_ids.add(item["id"])

        # Single commit for all rows
        try:
            db.commit()
            processed_count = len(work_items)
        except IntegrityError:
            db.rollback()
            # Fallback: commit one-by-one to salvage non-duplicate rows
            logger.warning("process_batch_bulk_commit_failed, falling back to row-by-row")
            topic_cache.clear()
            for (item, sent_idx, sentence, subject, _), (score, label) in zip(work_items, all_results):
                new_data_point = SentimentData(
                    source_id=f"{item['source_id']}_v{item['content_version']}_s{sent_idx}_{subject}",
                    source_type="topic_sentence",
                    sentiment_label=label,
                    sentiment_score=score,
                    relevant_text=sentence,
                )
                db.add(new_data_point)

                if subject not in topic_cache:
                    topic = db.query(Topic).filter(Topic.slug == subject).first()
                    if not topic:
                        topic = Topic(slug=subject, name=subject)
                        db.add(topic)
                    topic_cache[subject] = topic
                new_data_point.topics.append(topic_cache[subject])

                try:
                    db.commit()
                    processed_count += 1
                except IntegrityError:
                    db.rollback()
                    skipped_count += 1

        for item in claimed_rows:
            if item["id"] in processed_item_ids or failed_count == 0:
                mark_processed(item["id"])
    except Exception:
        logger.exception("process_batch_db_write_failed")
        db.rollback()
        failed_count += 1
    finally:
        db.close()

    result = {
        "claimed": len(claimed_rows),
        "processed": processed_count,
        "failed": failed_count,
        "skipped": skipped_count,
    }
    logger.info(
        "process_batch_completed claimed=%s processed=%s failed=%s skipped=%s",
        result["claimed"],
        result["processed"],
        result["failed"],
        result["skipped"],
    )
    if result["claimed"] > 0:
        failure_ratio = (result["failed"] + result["skipped"]) / result["claimed"]
        if failure_ratio >= 0.2:
            logger.warning(
                "process_batch_high_failure_ratio claimed=%s failed=%s skipped=%s ratio=%.3f",
                result["claimed"],
                result["failed"],
                result["skipped"],
                failure_ratio,
            )
    return result

