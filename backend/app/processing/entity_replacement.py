import re
from app.processing.sentiment_tagger.topic_definitions import SENTENCE_TOPIC_MAP
from app.processing.sentiment_tagger.tagger_logic import _GENERAL_TOPICS

ALL_STOCK_TICKERS = frozenset(
    t for t in SENTENCE_TOPIC_MAP if t not in _GENERAL_TOPICS
)

TARGET_TOKEN = "target"
OTHER_TOKEN = "other"

# Pre-compiled regex: matches any stock ticker in one pass.
# Sorted by length descending so longer tickers match first.
_TICKER_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(t) for t in sorted(ALL_STOCK_TICKERS, key=len, reverse=True)
    ) + r")\b"
)

def apply_entity_replacement(text: str, target_subject: str) -> str:
    """
    Replace the target subject ticker with "target" and all other known
    stock tickers with "other" in the normalized text (SEntFiN-style).

    The normalized_text from the DB already has company names replaced with
    uppercase tickers (e.g., "apple" → "AAPL") by normalize_and_tag_sentence().

    Example:
        text="AAPL is great but TSLA is doomed", subject="AAPL"
        → "target is great but other is doomed"
    """
    def _replacer(match: re.Match) -> str:
        return TARGET_TOKEN if match.group() == target_subject else OTHER_TOKEN

    return _TICKER_PATTERN.sub(_replacer, text)