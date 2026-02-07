from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import date  
from pydantic import BaseModel

from db.database import get_db_session
from db.models import Topic, SentimentData, WordFrequency  
from api.schemas import SentimentData as SentimentSchema

# 1. Create a router
router = APIRouter(
    prefix="/social-sentiment",
    tags=["Factor: Social Sentiment"]
)

# 2. Define your data schemas (Pydantic models)
class DailySentiment(BaseModel):
    day: date
    average_score: float

class WordCloudItem(BaseModel):
    text: str
    value: int
    class Config:
        from_attributes = True

# --- Endpoints for Ticker Search ---

@router.get("/{ticker}/daily", response_model=List[DailySentiment])
async def get_daily_sentiment_for_ticker(ticker: str, db: Session = Depends(get_db_session)):
    """
    Get the average sentiment score per day for a specific ticker's chart.
    """
    topic = db.query(Topic).filter(Topic.slug == ticker.upper()).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    daily_data = (
        db.query(
            func.date_trunc('day', SentimentData.created_at).label('day'),
            func.avg(SentimentData.sentiment_score).label('average_score')
        )
        .filter(SentimentData.topics.contains(topic))
        .group_by('day')
        .order_by('day')
        .all()
    )
    return daily_data

@router.get("/{ticker}/evidence", response_model=List[SentimentSchema])
async def get_sentiment_evidence_for_ticker(ticker: str, db: Session = Depends(get_db_session)):
    """
    Get the 5 most recent sentiment records (for "evidence").
    """
    topic = db.query(Topic).filter(Topic.slug == ticker.upper()).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    recent_evidence = (
        db.query(SentimentData)
        .filter(SentimentData.topics.contains(topic))
        .order_by(SentimentData.created_at.desc())
        .limit(5)  # <-- Changed to 5 to match your requirement
        .all()
    )
    return recent_evidence

# --- NEW Endpoints for Summary Page ---

@router.get("/summary/{topic_slug}", response_model=DailySentiment)
async def get_topic_summary(topic_slug: str, db: Session = Depends(get_db_session)):
    """
    Get the single, average sentiment for a broad topic (like 'MACRO') for today.
    """
    topic = db.query(Topic).filter(Topic.slug == topic_slug.upper()).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    result = (
        db.query(func.avg(SentimentData.sentiment_score).label('average_score'))
        .filter(SentimentData.topics.contains(topic))
        .first()
    )
    
    return {"day": date.today(), "average_score": result.average_score or 0.0}

@router.get("/wordcloud", response_model=List[WordCloudItem])
async def get_word_cloud(db: Session = Depends(get_db_session)):
    """
    Get the top 100 most frequent words for today.
    """
    word_data = (
        db.query(WordFrequency.word, WordFrequency.frequency)
        # .filter(WordFrequency.date == date.today())
        .order_by(WordFrequency.frequency.desc())
        .limit(100) # <-- Set to 100 as you requested
        .all()
    )
    
    # Convert data to the format: [{text: "word", value: 10}]
    return [{"text": word, "value": freq} for word, freq in word_data]