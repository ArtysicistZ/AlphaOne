import re
from .topic_definitions import SENTENCE_TOPIC_MAP

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
            # We check the pre-lowercased sentence
            if keyword in sentence_lower:
                found_topics.add(topic_slug)
                break # Move to the next topic (no need to check other keywords)
            
    return found_topics