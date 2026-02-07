from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db_session
from db.models import Topic
from api.schemas import Topic as TopicSchema

# 1. Create a router
router = APIRouter(
    prefix="/assets",  # This adds "/assets" to all URLs in this file
    tags=["Assets"]    # This groups them in the auto-docs
)

@router.get("/tracked", response_model=List[TopicSchema])
async def get_tracked_assets(db: Session = Depends(get_db_session)):
    """
    Get a list of all tracked assets (e.g., NVDA, AAPL)
    that have sentiment data.
    """
    topics = db.query(Topic).all()
    return topics