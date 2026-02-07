# in /core/nlp_processor.py
# This module provides NLP processing functions, specifically sentiment analysis.
# It uses a transformer model for sentiment classification.

# _clean_text(text: str) -> str
# get_sentiment(text: str) -> dict  

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- 1. Load the Tokenizer and Model Manually ---
MODEL_NAME = "ProsusAI/finbert"

print("Loading FinBERT model and tokenizer...")
try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    MODEL.eval() # Put the model in evaluation mode
except OSError:
    print(f"\n[Error] Could not download FinBERT model ('{MODEL_NAME}').")
    print("Please check your internet connection.\n")
    exit()
print("FinBERT model loaded successfully.")


def _softmax(x):
    """Compute softmax values for a one-dimensional array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_sentiment(text: str) -> (float, str):
    """
    Performs a full, statistically accurate sentiment analysis.
    
    Returns:
        (final_score, label) 
        e.g., (0.85, "positive") or (-0.5, "negative")
    """
    if not text:
        return (0.0, 'neutral')

    try:
        # --- 2. Tokenize the text ---
        inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # --- 3. Get Model Output (Logits) ---
        with torch.no_grad():
            outputs = MODEL(**inputs)
        
        # 'logits' are the raw, un-normalized scores for each label
        logits = outputs.logits[0].numpy()
        
        # --- 4. Get Probabilities ---
        # Apply softmax to convert logits to probabilities (0% to 100%)
        probabilities = _softmax(logits)
        
        # FinBERT labels are: [0] = negative, [1] = neutral, [2] = positive
        prob_negative = probabilities[0]
        prob_neutral = probabilities[1]
        prob_positive = probabilities[2]
        
        # --- 5. Apply Your "More Accurate Function" ---
        # Score = (Probability of Positive) - (Probability of Negative)
        final_score = prob_positive - prob_negative
        
        # Determine the final label based on the score
        if final_score > 0.05:
            final_label = 'positive'
        elif final_score < -0.05:
            final_label = 'negative'
        else:
            final_label = 'neutral'

        return (float(final_score), final_label)

    except Exception as e:
        print(f"Error in NLP processing: {e}")
        return (0.0, 'neutral')