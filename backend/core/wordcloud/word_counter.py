import re
from collections import Counter
from .stop_words import STOP_WORDS
from .keyword_map import KEYWORD_MAP  # <-- 1. Import your new map

def get_word_counts(text: str) -> Counter:
    """
    Cleans a string, applies normalization and weighting from the
    keyword_map, removes stop words, and returns a Counter object.
    """
    if not isinstance(text, str):
        return Counter()
    
    weighted_counts = Counter()
    
    # 1. Clean and split the text
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # 2. Iterate and apply weights/normalization
    for word in words:
        
        # Check if it's a weighted keyword (e.g., 'aapl')
        if word in KEYWORD_MAP:
            standard_name, weight = KEYWORD_MAP[word]
            weighted_counts[standard_name] += weight
            
        # Check if it's a "boring" stop word
        elif word in STOP_WORDS:
            continue
            
        # Check for simple junk words
        elif len(word) < 3 or len(word) > 20:
            continue
            
        # If it's a normal, non-boring word
        else:
            weighted_counts[word] += 1.0 # Add a normal weight of 1
            
    return weighted_counts