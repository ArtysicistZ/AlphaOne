from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


MODEL_NAME = "ProsusAI/finbert"
_TOKENIZER = None
_MODEL = None


def _get_finbert():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _MODEL.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load FinBERT model '{MODEL_NAME}'") from exc
    return _TOKENIZER, _MODEL


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_sentiment(text: str) -> tuple[float, str]:
    if not text:
        return (0.0, "neutral")

    try:
        tokenizer, model = _get_finbert()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0].numpy()
        probabilities = _softmax(logits)

        prob_negative = probabilities[0]
        prob_positive = probabilities[2]
        final_score = prob_positive - prob_negative

        if final_score > 0.05:
            final_label = "positive"
        elif final_score < -0.05:
            final_label = "negative"
        else:
            final_label = "neutral"

        return (float(final_score), final_label)
    except Exception:
        return (0.0, "neutral")

