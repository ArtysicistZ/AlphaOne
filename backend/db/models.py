# In db/models.py
# This module defines the database models using SQLAlchemy ORM.
# It includes a many-to-many relationship between SentimentData and Topics.

# class Topic(Base)
# class SentimentData(Base)


from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Table, ForeignKey
from sqlalchemy import Date, func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.schema import UniqueConstraint

Base = declarative_base()

# --- Association Table ---
# This special table links SentimentData and Topics
# It has no Python class. It just defines the relationship.
# Association tables do not need their own class.
sentiment_topic_association = Table(
    'sentiment_topic_association', 
    Base.metadata,
    Column('sentiment_data_id', 
           Integer, 
           ForeignKey('sentiment_data.id', ondelete="CASCADE"), # <-- ADD THIS
           primary_key=True),
    Column('topic_id', 
           Integer, 
           ForeignKey('topics.id'), 
           primary_key=True)
)

# --- Topic Table ---
# This will store our "catalogs" (NVDA, AAPL, etc.)
class Topic(Base):
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True) # e.g., "NVDA"
    name = Column(String) # e.g., "Nvidia"
    
    # This defines the "many" side of the many-to-many
    sentiments = relationship(
        "SentimentData",
        secondary=sentiment_topic_association,
        back_populates="topics"
    )

# --- Sentiment Data Table ---
class SentimentData(Base):
    __tablename__ = "sentiment_data"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    source_id = Column(String, unique=True, index=True)
    source_type = Column(String)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    relevant_text = Column(String, nullable=True) # To store the specific sentences

    # --- This relationship stays the same ---
    topics = relationship(
        "Topic",
        secondary=sentiment_topic_association,
        back_populates="sentiments"
    )

class WordFrequency(Base):
    __tablename__ = "word_frequency"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, index=True, nullable=False)
    frequency = Column(Integer, nullable=False)
    date = Column(Date, index=True, nullable=False, server_default=func.current_date())
    
    # This ensures we don't have duplicate entries for the same word on the same day
    __table_args__ = (UniqueConstraint('word', 'date', name='_word_date_uc'),)