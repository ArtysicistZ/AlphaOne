from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Optional

import praw

from app.settings import CLIENT_ID, SECRET_KEY, USERNAME, PASSWORD


class RedditClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        user_agent: str = "alphaone-reddit-worker/1.0",
    ) -> None:
        self.client_id = client_id or CLIENT_ID
        self.client_secret = client_secret or SECRET_KEY
        self.username = username or USERNAME
        self.password = password or PASSWORD
        self.user_agent = user_agent
        self._reddit: Optional[praw.Reddit] = None

    def _get_client(self) -> praw.Reddit:
        if self._reddit is None:
            if not all([self.client_id, self.client_secret, self.username, self.password]):
                raise ValueError("Missing Reddit credentials. Check .env and app/settings.py.")
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent,
            )
        return self._reddit

    @staticmethod
    def _to_utc(ts: Optional[float]) -> Optional[datetime]:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    
    @staticmethod
    def _edited_to_utc(edited_value) -> Optional[datetime]:
        if not edited_value or edited_value is True:
            return None
        if isinstance(edited_value, (int, float)):
            return datetime.fromtimestamp(edited_value, tz=timezone.utc)
        return None

    @staticmethod
    def _normalize_comment(comment, subreddit_name: str) -> Optional[Dict]:
        text = (comment.body or "").strip()
        if not text:
            return None
        return {
            "source_id": f"reddit_comment:{comment.id}",
            "source_type": "reddit_comment",
            "text": text,
            "subreddit": subreddit_name,
            "created_utc": RedditClient._to_utc(getattr(comment, "created_utc", None)),
            "edited_utc": RedditClient._edited_to_utc(getattr(comment, "edited", None)),
        }

    @staticmethod
    def _normalize_submission(submission, subreddit_name: str) -> Optional[Dict]:
        title = (submission.title or "").strip()
        body = (submission.selftext or "").strip()
        text = f"{title}\n{body}".strip()
        if not text:
            return None
        return {
            "source_id": f"reddit_submission:{submission.id}",
            "source_type": "reddit_submission",
            "text": text,
            "subreddit": subreddit_name,
            "created_utc": RedditClient._to_utc(getattr(submission, "created_utc", None)),
            "edited_utc": RedditClient._edited_to_utc(getattr(submission, "edited", None)),
        }

    def fetch_raw_rows(
        self,
        subreddits: List[str],
        limit: int = 100,
        include_comments: bool = True,
        include_submissions: bool = True,
    ) -> List[Dict]:
        reddit = self._get_client()
        rows: List[Dict] = []

        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)

            if include_comments:
                for comment in subreddit.comments(limit=limit):
                    row = self._normalize_comment(comment, subreddit_name)
                    if row:
                        rows.append(row)

            if include_submissions:
                for submission in subreddit.hot(limit=limit):
                    row = self._normalize_submission(submission, subreddit_name)
                    if row:
                        rows.append(row)

        return rows




