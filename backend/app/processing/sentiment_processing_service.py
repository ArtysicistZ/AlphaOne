from collections import Counter
from datetime import date
import logging

from sqlalchemy.exc import IntegrityError

from app.processing.nlp_processor import get_sentiment
from app.processing.wordcloud.word_counter import get_word_counts
from app.processing.text_utils import split_into_sentences
from app.processing.sentiment_tagger.tagger_logic import get_topics_from_sentence, normalize_and_tag_sentence, MAX_TICKERS_PER_SENTENCE
from app.ingestion.raw_ingest_service import claim_new_raw_posts, mark_processed, mark_failed
from app.database.session import SessionLocal, init_db
from app.database.models import SentimentData, Topic, WordFrequency, ProcessedSentence

logger = logging.getLogger(__name__)


def process_batch(limit: int = 100):
    init_db()
    logger.info("process_batch_started limit=%s", limit)
    claimed_rows = claim_new_raw_posts(limit=limit)
    if not claimed_rows:
        logger.info("No new raw posts to process. Exiting batch.")
        return {"claimed": 0, "processed": 0, "failed": 0, "skipped": 0}

    db = SessionLocal()
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    total_word_counts = Counter()

    try:
        for item in claimed_rows:
            try:
                full_text = item["text"]
                if not full_text or not isinstance(full_text, str):
                    mark_failed(item["id"])
                    failed_count += 1
                    continue

                total_word_counts.update(get_word_counts(full_text))
                sentences = split_into_sentences(full_text)

                for i, sentence in enumerate(sentences):
                    topics_in_sentence = get_topics_from_sentence(sentence)
                    if not topics_in_sentence:
                        continue
                    if len(topics_in_sentence) > MAX_TICKERS_PER_SENTENCE:
                        continue

                    final_score, final_label = get_sentiment(sentence)
                    new_data_point = SentimentData(
                        source_id=f"{item['source_id']}_v{item['content_version']}_s{i}",
                        source_type="topic_sentence",
                        sentiment_label=final_label,
                        sentiment_score=final_score,
                        relevant_text=sentence,
                    )
                    db.add(new_data_point)

                    for topic_slug in topics_in_sentence:
                        topic = db.query(Topic).filter_by(slug=topic_slug).first()
                        if not topic:
                            topic = Topic(slug=topic_slug, name=topic_slug)
                            db.add(topic)
                        new_data_point.topics.append(topic)

                    try:
                        db.commit()
                        processed_count += 1
                    except IntegrityError:
                        db.rollback()
                        skipped_count += 1

                mark_processed(item["id"])
            except Exception:
                logger.exception("Error processing item %s", item.get("id", "N/A"))
                db.rollback()
                mark_failed(item["id"])
                failed_count += 1

        top_100_words = total_word_counts.most_common(100)
        today = date.today()
        try:
            db.query(WordFrequency).filter(WordFrequency.date == today).delete()
            for word, frequency in top_100_words:
                db.add(WordFrequency(word=word, frequency=int(frequency), date=today))
            db.commit()
        except Exception:
            logger.exception("Error saving word batch to DB")
            db.rollback()
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


def reprocess_batch(limit: int = 100):
    """
    Process raw posts into the processed_sentences table.
    Each (sentence, subject) pair gets its own row with an explicit subject column.
    """
    init_db()
    logger.info("reprocess_batch_started limit=%s", limit)
    claimed_rows = claim_new_raw_posts(limit=limit)
    if not claimed_rows:
        logger.info("No new raw posts to reprocess. Exiting batch.")
        return {"claimed": 0, "processed": 0, "failed": 0, "skipped": 0}

    db = SessionLocal()
    processed_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for item in claimed_rows:
            try:
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

                    final_score, final_label = get_sentiment(sentence)

                    for subject in topics_in_sentence:
                        row = ProcessedSentence(
                            raw_post_id=item["id"],
                            source_id=f"{item['source_id']}_v{item['content_version']}_s{i}__{subject}",
                            sentence_index=i,
                            sentence_text=normalized_text,
                            subject=subject,
                            sentiment_label=final_label,
                            sentiment_score=final_score,
                            subreddit=item.get("subreddit"),
                            created_utc=item.get("created_utc"),
                        )
                        db.add(row)

                        try:
                            db.commit()
                            processed_count += 1
                        except IntegrityError:
                            db.rollback()
                            skipped_count += 1

                mark_processed(item["id"])
            except Exception:
                logger.exception("Error reprocessing item %s", item.get("id", "N/A"))
                db.rollback()
                mark_failed(item["id"])
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
        "reprocess_batch_completed claimed=%s processed=%s failed=%s skipped=%s",
        result["claimed"],
        result["processed"],
        result["failed"],
        result["skipped"],
    )
    return result
