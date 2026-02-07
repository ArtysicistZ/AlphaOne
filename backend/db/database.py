# In db/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base  # Import the Base from models.py
from config import DATABASE_URL  # Import your database connection string

engine = create_engine(
    DATABASE_URL,
    # Neon connection optimization:
    # Neon may put idle computes to sleep. This checks the connection's health
    # before use, preventing sudden disconnect errors in a serverless environment.
    pool_pre_ping=True 
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # This command creates all the tables defined in models.py
    Base.metadata.create_all(bind=engine)

def get_db_session():
    """Helper function to get a new database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()