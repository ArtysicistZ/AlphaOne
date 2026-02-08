from sqlalchemy import create_engine, text
import logging

from app.settings import DATABASE_URL

logger = logging.getLogger(__name__)


def cleanup_old_data():
    logger.info("Connecting to database to clean up old data...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            query_sentiment = text(
                """
                DELETE FROM sentiment_data
                WHERE created_at < NOW() - INTERVAL '7 days'
                """
            )
            result_sentiment = connection.execute(query_sentiment)

            query_words = text(
                """
                DELETE FROM word_frequency
                WHERE date < NOW() - INTERVAL '7 days'
                """
            )
            result_words = connection.execute(query_words)
            connection.commit()

            logger.info(
                "Cleanup complete. Deleted %s old sentiment records.",
                result_sentiment.rowcount,
            )
            logger.info(
                "Deleted %s old word frequency records.",
                result_words.rowcount,
            )
    except Exception:
        logger.exception("An error occurred during cleanup")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_old_data()



