# in /backend/batch_processor.py
# This module handles batch processing of Reddit comment data.
# It reads raw data files, processes them using NLP functions,
# and loads the results into a database. It also counts word frequencies across all data.

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import pandas as pd
import json
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from collections import Counter
from datetime import date

# --- Core Imports ---
from core.nlp_processor import get_sentiment
from core.wordcloud.word_counter import get_word_counts

# --- MODIFIED: Import your new, correct functions ---
from core.text_utils import split_into_sentences
from core.sentiment_tagger.tagger_logic import get_topics_from_sentence
# ---

# --- DB Imports ---
from db.database import SessionLocal, init_db
from db.models import SentimentData, Topic, WordFrequency

def process_batch():
    """
    Reads all raw data, processes each post SENTENCE-BY-SENTENCE,
    and calculates word frequencies.
    """
    
    # --- Setup ---
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "reddit" / "raw"
    
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path.absolute()}")
        return

    init_db() 
    db = SessionLocal()
    print("Database connection established.")

    processed_count = 0
    skipped_count = 0
    total_word_counts = Counter()
    
    try:
        print(f"Starting batch process for files in {data_path}...")
        for file_path in data_path.glob('*'):
            
            if not file_path.is_file(): continue
            data_to_process = []
            
            # --- (Your file-reading logic is unchanged and correct) ---
            if file_path.suffix == '.csv':
                print(f"Reading CSV: {file_path.name}")
                df = pd.read_csv(file_path)
                if 'comment_id' in df.columns and 'body' in df.columns:
                    for _, row in df.iterrows():
                        data_to_process.append({'id': row['comment_id'], 'text': row['body']})
                elif 'submission_id' in df.columns and 'title' in df.columns:
                    for _, row in df.iterrows():
                        text = f"{row['title']} {row.get('selftext', '')}"
                        data_to_process.append({'id': row['submission_id'], 'text': text})
            elif file_path.suffix == '.json':
                print(f"Reading JSON: {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if 'comment_id' in item and 'body' in item:
                            data_to_process.append({'id': item['comment_id'], 'text': item['body']})
                        elif 'submission_id' in item and 'title' in item:
                            text = f"{item['title']} {item.get('selftext', '')}"
                            data_to_process.append({'id': item['submission_id'], 'text': text})
            else:
                print(f"Skipping unsupported file: {file_path.name}")
                continue
            if not data_to_process:
                print(f"No valid data found in {file_path.name}")
                continue

            print(f"Processing {len(data_to_process)} items from {file_path.name}...")

            # --- Main Processing Loop ---
            for item in data_to_process:
                try:
                    full_text = item.get('text', '')
                    if not full_text or not isinstance(full_text, str):
                        continue

                    # --- 1. Word Cloud Logic (Runs on full text) ---
                    total_word_counts.update(get_word_counts(full_text))

                    # --- 2. MODIFIED: NEW SENTENCE-BY-SENTENCE LOGIC ---
                    
                    # A. Use your spaCy-powered splitter to get a list of clean sentences
                    sentences = split_into_sentences(full_text) 

                    # B. Iterate over each *individual* clean sentence
                    for i, sentence in enumerate(sentences):
                        
                        # C. Find topics in this one sentence
                        topics_in_sentence = get_topics_from_sentence(sentence)
                        
                        # D. If no topics, skip this sentence
                        if not topics_in_sentence:
                            continue
                            
                        # E. If topics ARE found, run NLP *only on this sentence*
                        final_score, final_label = get_sentiment(sentence)
                        
                        # F. Create the DB object with the single sentence
                        new_data_point = SentimentData(
                            source_id=f"{item['id']}_s{i}", # Unique ID per sentence
                            source_type='topic_sentence',
                            sentiment_label=final_label,
                            sentiment_score=final_score, # This is the true polarity score
                            relevant_text=sentence # Store only the single, clean sentence
                        )
                        
                        # G. Add the new data point to the session *first*
                        db.add(new_data_point)
                        
                        # H. Find/create and link all topics found in this sentence
                        for topic_slug in topics_in_sentence:
                            topic = db.query(Topic).filter_by(slug=topic_slug).first()
                            if not topic:
                                topic = Topic(slug=topic_slug, name=topic_slug)
                                db.add(topic)
                            
                            new_data_point.topics.append(topic)
                        
                        # I. Commit all changes (the data point, new topics, and links)
                        try:
                            db.commit()
                            processed_count += 1
                        except IntegrityError:
                            db.rollback()
                            skipped_count += 1
                        except Exception as e:
                            print(f"Error saving to DB: {e}")
                            db.rollback()
                
                except Exception as e:
                    print(f"Error processing item {item.get('id', 'N/A')}: {e}")
                    db.rollback()

        # --- END OF ALL FILE LOOPS ---
        
        # --- (Your "Save Top 100 Words" logic is unchanged and correct) ---
        print(f"\nAll files processed. Aggregating and saving top 100 words for {date.today()}...")
        top_100_words = total_word_counts.most_common(100)
        today = date.today()
        
        try:
            db.query(WordFrequency).filter(WordFrequency.date == today).delete()
            print(f"Cleared all existing word data for {today}.")
            for word, frequency in top_100_words:
                new_word = WordFrequency(word=word, frequency=int(frequency), date=today)
                db.add(new_word)
            db.commit()
            print("Top 100 words saved successfully.")
        except Exception as e:
            print(f"Error saving word batch to DB: {e}")
            db.rollback()

    finally:
        db.close()
        
    print("--- Batch Processing Complete ---")
    print(f"New records added: {processed_count}")
    print(f"Duplicate records skipped: {skipped_count}")


if __name__ == "__main__":
    process_batch()