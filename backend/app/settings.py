import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
SECRET_KEY = os.getenv("REDDIT_SECRET_KEY")
USERNAME = os.getenv("REDDIT_USERNAME")
PASSWORD = os.getenv("REDDIT_PASSWORD")

REDDIT_SUBREDDITS = [
    s.strip()
    for s in os.getenv("REDDIT_SUBREDDITS", "wallstreetbets,stocks,investing").split(",")
    if s.strip()
]

REDDIT_FETCH_LIMIT = int(os.getenv("REDDIT_FETCH_LIMIT", "100"))
BATCH_PROCESS_LIMIT = int(os.getenv("BATCH_PROCESS_LIMIT", "100"))
