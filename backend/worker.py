# In worker.py (or batch_processor.py)

import sys
import os
# Add the project root directory (the parent of this script) to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


from core.reddit_client import fetch_comments_from_subreddit
from core.nlp_processor import get_sentiment
from core.topic_tagger import get_topics 
from db.database import SessionLocal, init_db
from db.models import SentimentData, Topic  
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

def run_pipeline():
    init_db() # This will now create all 3 tables
    db: Session = next(SessionLocal())
    
    print("Fetching raw data from Reddit...")
    raw_comments = fetch_comments_from_subreddit("wallstreetbets", limit=100)
    
    new_records_added = 0
    for comment in raw_comments:
        try:
            # --- TRANSFORM (Step 1 & 2) ---
            sentiment_result = get_sentiment(comment.body)
            topic_slugs = get_topics(comment.body) # <-- CALL THE TAGGER
            
            # --- LOAD (Step 1: Create SentimentData) ---
            new_data_point = SentimentData(
                source_id=comment.id,
                source_type='comment',
                text_content=comment.body,
                sentiment_label=sentiment_result['label'],
                sentiment_score=sentiment_result['score']
            )
            
            # --- LOAD (Step 2: Find or Create Topics) ---
            for slug in topic_slugs:
                # Check if topic exists
                topic = db.query(Topic).filter_by(slug=slug).first()
                if not topic:
                    # If not, create it
                    topic = Topic(slug=slug, name=slug) # You can add a full name later
                    db.add(topic)
                    db.flush() # Use flush to get the ID before committing
                
                # Add the relationship
                new_data_point.topics.append(topic)
            
            # --- LOAD (Step 3: Commit to DB) ---
            db.add(new_data_point)
            db.commit()
            new_records_added += 1

        except IntegrityError:
            # This correctly catches duplicate source_id
            db.rollback()
        except Exception as e:
            print(f"Error processing item {comment.id}: {e}")
            db.rollback()

    print(f"Pipeline run complete. Added {new_records_added} new records.")
    db.close()

if __name__ == "__main__":
    run_pipeline()