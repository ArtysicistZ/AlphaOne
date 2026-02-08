from collections import Counter
from datetime import date

from sqlalchemy.exc import IntegrityError

from app.processing.nlp_processor import get_sentiment
from app.processing.wordcloud.word_counter import get_word_counts
from app.processing.text_utils import split_into_sentences
from app.processing.sentiment_tagger.tagger_logic import get_topics_from_sentence
from app.ingestion.raw_ingest_service import claim_new_raw_posts, mark_processed, mark_failed
from app.database.session import SessionLocal, init_db
from app.database.models import SentimentData, Topic, WordFrequency


def process_batch(limit: int = 100):
    init_db()
    claimed_rows = claim_new_raw_posts(limit=limit)
    if not claimed_rows:
        print("No new raw posts to process. Exiting batch.")
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

                    final_score, final_label = get_sentiment(sentence)
                    new_data_point = SentimentData(
                        source_id=f"{item['source_id']}_s{i}",
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
            except Exception as exc:
                print(f"Error processing item {item.get('id', 'N/A')}: {exc}")
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
        except Exception as exc:
            print(f"Error saving word batch to DB: {exc}")
            db.rollback()
    finally:
        db.close()

    return {
        "claimed": len(claimed_rows),
        "processed": processed_count,
        "failed": failed_count,
        "skipped": skipped_count,
    }



