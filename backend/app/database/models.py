from sqlalchemy import Column, Integer, String, Float, DateTime, Table, ForeignKey, Text
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


class ProcessedSentence(Base):
    __tablename__ = "processed_sentences"

    id = Column(Integer, primary_key=True, index=True)
    raw_post_id = Column(Integer, ForeignKey("raw_reddit_posts.id"), nullable=False)
    source_id = Column(String, unique=True, index=True)
    sentence_index = Column(Integer, nullable=False)
    sentence_text = Column(String, nullable=False)
    subject = Column(String, nullable=False, index=True)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    subreddit = Column(String, nullable=True)
    created_utc = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("raw_post_id", "sentence_index", "subject", name="_raw_sent_subj_uc"),
    )


# ── 3NF Training Tables ──────────────────────────────────────────────────
# Sentence text is stored once; subjects are in a separate child table.


class TrainingSentence(Base):
    __tablename__ = "training_sentences"

    id = Column(Integer, primary_key=True, index=True)
    raw_post_id = Column(Integer, ForeignKey("raw_reddit_posts.id"), nullable=False)
    sentence_index = Column(Integer, nullable=False)
    normalized_text = Column(Text, nullable=False)
    subreddit = Column(String, nullable=True)
    created_utc = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

    subjects = relationship("TrainingSentenceSubject", back_populates="sentence", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("raw_post_id", "sentence_index", name="_train_raw_sent_uc"),
    )


class TrainingSentenceSubject(Base):
    __tablename__ = "training_sentence_subjects"

    id = Column(Integer, primary_key=True, index=True)
    sentence_id = Column(Integer, ForeignKey("training_sentences.id", ondelete="CASCADE"), nullable=False)
    subject = Column(String, nullable=False, index=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_confidence = Column(Float, nullable=True)

    sentence = relationship("TrainingSentence", back_populates="subjects")

    __table_args__ = (
        UniqueConstraint("sentence_id", "subject", name="_train_sent_subj_uc"),
    )

