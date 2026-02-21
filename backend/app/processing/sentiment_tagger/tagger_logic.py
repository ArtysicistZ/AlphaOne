import re
from .topic_definitions import SENTENCE_TOPIC_MAP

# General topics where keyword replacement doesn't make sense
# (replacing "inflation" with "MACRO" would destroy meaning)
_GENERAL_TOPICS = frozenset({"MACRO", "TECHNOLOGY"})

# Sentences mentioning too many tickers are portfolio lists / tables —
# no meaningful per-stock sentiment signal.
MAX_TICKERS_PER_SENTENCE = 7


def get_topics_from_sentence(sentence: str) -> set[str]:
    """
    Checks a single, clean sentence and returns a set of all topic
    slugs (e.g., 'AAPL', 'NVDA') found in it.
    """
    found_topics = set()

    # 1. We lowercase the sentence for matching
    sentence_lower = sentence.lower()

    # 2. Check against our topic map
    for topic_slug, keywords in SENTENCE_TOPIC_MAP.items():
        for keyword in keywords:
            if keyword.startswith("$"):
                # $-prefix: anchor end with word boundary so $ms doesn't match $msft
                matched = bool(re.search(re.escape(keyword) + r"\b", sentence_lower))
            else:
                matched = bool(re.search(
                    r"\b" + re.escape(keyword) + r"\b", sentence_lower
                ))
            if matched:
                found_topics.add(topic_slug)
                break  # Move to the next topic

    return found_topics


def normalize_and_tag_sentence(sentence: str) -> tuple[str, set[str]]:
    """
    Detects topics AND normalizes the sentence by replacing matched keywords
    with their canonical ticker symbol.

    Example:
        "apple earnings were great"  -> ("AAPL earnings were great", {"AAPL"})
        "$tsla to the moon"          -> ("TSLA to the moon",        {"TSLA"})
        "fed raised rates"           -> ("fed raised rates",        {"MACRO"})
                                         ^ general topics not replaced

    Returns (normalized_sentence, found_topics).
    """
    found_topics: set[str] = set()
    replacements: list[tuple[str, str]] = []  # (matched_keyword, ticker)
    sentence_lower = sentence.lower()

    for topic_slug, keywords in SENTENCE_TOPIC_MAP.items():
        if topic_slug in _GENERAL_TOPICS:
            # Detect only — don't replace "inflation" with "MACRO"
            for keyword in keywords:
                if bool(re.search(r"\b" + re.escape(keyword) + r"\b", sentence_lower)):
                    found_topics.add(topic_slug)
                    break
            continue

        # Find the longest matching keyword for this topic so that
        # "berkshire hathaway" is preferred over "berkshire"
        best_keyword = None
        for keyword in keywords:
            if keyword.startswith("$"):
                # Anchor end with word boundary so $ms doesn't match $msft
                matched = bool(re.search(re.escape(keyword) + r"\b", sentence_lower))
            else:
                matched = bool(re.search(
                    r"\b" + re.escape(keyword) + r"\b", sentence_lower
                ))
            if matched and (best_keyword is None or len(keyword) > len(best_keyword)):
                best_keyword = keyword

        if best_keyword:
            found_topics.add(topic_slug)
            replacements.append((best_keyword, topic_slug))

    # Replace longer keywords first to avoid partial clobbering
    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    normalized = sentence
    for keyword, ticker in replacements:
        if keyword.startswith("$"):
            pattern = re.escape(keyword) + r"\b"
        else:
            pattern = r"\b" + re.escape(keyword) + r"\b"
        normalized = re.sub(pattern, ticker, normalized, flags=re.IGNORECASE)

    return normalized, found_topics