from pydantic import BaseModel
from datetime import datetime

# This is what a SINGLE sentiment record should look like
class SentimentData(BaseModel):
    id: int
    created_at: datetime
    sentiment_label: str
    sentiment_score: float
    relevant_text: str | None = None # This field is optional

    class Config:
        from_attributes = True

# This is what a SINGLE topic should look like
class Topic(BaseModel):
    id: int
    slug: str

    class Config:
        from_attributes = True