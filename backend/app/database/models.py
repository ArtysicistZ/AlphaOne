from sqlalchemy import Column, Integer, String, Float, DateTime, Table, ForeignKey
from sqlalchemy import Date, func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.schema import UniqueConstraint


Base = declarative_base()


sentiment_topic_association = Table(
    "sentiment_topic_association",
    Base.metadata,
    Column(
        "sentiment_data_id",
        Integer,
        ForeignKey("sentiment_data.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("topic_id", Integer, ForeignKey("topics.id"), primary_key=True),
)


class Topic(Base):
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True)
    name = Column(String)

    sentiments = relationship(
        "SentimentData",
        secondary=sentiment_topic_association,
        back_populates="topics",
    )


class SentimentData(Base):
    __tablename__ = "sentiment_data"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    source_id = Column(String, unique=True, index=True)
    source_type = Column(String)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    relevant_text = Column(String, nullable=True)

    topics = relationship(
        "Topic",
        secondary=sentiment_topic_association,
        back_populates="sentiments",
    )


class WordFrequency(Base):
    __tablename__ = "word_frequency"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, index=True, nullable=False)
    frequency = Column(Integer, nullable=False)
    date = Column(Date, index=True, nullable=False, server_default=func.current_date())

    __table_args__ = (UniqueConstraint("word", "date", name="_word_date_uc"),)


class RawRedditPost(Base):
    __tablename__ = "raw_reddit_posts"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String, unique=True, index=True)
    source_type = Column(String)
    text = Column(String)
    subreddit = Column(String, nullable=True)
    created_utc = Column(DateTime(timezone=True), nullable=True)
    edited_utc = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, default="new")
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    fetch_count = Column(Integer, default=1)
    content_version = Column(Integer, nullable=False, default=1)

