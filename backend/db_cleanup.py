import sys
import os
# Add the project root directory (the parent of this script) to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from sqlalchemy import create_engine, text
from db.database import DATABASE_URL 

def cleanup_old_data():
    print("Connecting to database to clean up old data...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            
            # --- 1. Your existing cleanup query ---
            query_sentiment = text("""
                DELETE FROM sentiment_data 
                WHERE created_at < NOW() - INTERVAL '7 days'
            """)
            result_sentiment = connection.execute(query_sentiment)
            
            # --- 2. ADD THIS NEW QUERY ---
            query_words = text("""
                DELETE FROM word_frequency 
                WHERE date < NOW() - INTERVAL '7 days'
            """)
            result_words = connection.execute(query_words)
            
            connection.commit() # Commit both deletions
            
            print(f"Cleanup complete. Deleted {result_sentiment.rowcount} old sentiment records.")
            print(f"Deleted {result_words.rowcount} old word frequency records.")
            
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")

if __name__ == "__main__":
    cleanup_old_data()