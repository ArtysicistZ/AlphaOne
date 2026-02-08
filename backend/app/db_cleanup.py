from sqlalchemy import create_engine, text

from app.settings import DATABASE_URL


def cleanup_old_data():
    print("Connecting to database to clean up old data...")
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

            print(f"Cleanup complete. Deleted {result_sentiment.rowcount} old sentiment records.")
            print(f"Deleted {result_words.rowcount} old word frequency records.")
    except Exception as exc:
        print(f"An error occurred during cleanup: {exc}")


if __name__ == "__main__":
    cleanup_old_data()



