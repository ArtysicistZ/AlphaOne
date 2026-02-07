import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The database URL is now available for use everywhere
DATABASE_URL = os.getenv("DATABASE_URL")
CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
SECRET_KEY = os.getenv('REDDIT_SECRET_KEY')
USERNAME = os.getenv('REDDIT_USERNAME')
PASSWORD = os.getenv('REDDIT_PASSWORD')