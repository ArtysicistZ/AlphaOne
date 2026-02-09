from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging


MODEL_NAME = "ProsusAI/finbert"
_TOKENIZER = None
_MODEL = None
logger = logging.getLogger(__name__)


def _get_finbert():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            _MODEL.eval()
        except Exception as exc:
            logger.exception("finbert_load_failed model=%s", MODEL_NAME)
            raise RuntimeError(f"Failed to load FinBERT model '{MODEL_NAME}'") from exc
    return _TOKENIZER, _MODEL


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def _resolve_label_indices(model) -> tuple[int, int]:
    """
    Resolve positive/negative class indices from model metadata.
    Falls back to FinBERT defaults if metadata is missing.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    normalized = {}
    for idx, label in id2label.items():
        try:
            normalized[int(idx)] = str(label).lower()
        except Exception:
            continue

    pos_idx = next((idx for idx, label in normalized.items() if "positive" in label), None)
    neg_idx = next((idx for idx, label in normalized.items() if "negative" in label), None)

    # FinBERT common ordering is [positive, negative, neutral]
    if pos_idx is None:
        pos_idx = 0
    if neg_idx is None:
        neg_idx = 1
    return pos_idx, neg_idx


def get_sentiment(text: str) -> tuple[float, str]:
    if not text:
        return (0.0, "neutral")

    try:
        tokenizer, model = _get_finbert()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0].detach().cpu().numpy()
        probabilities = _softmax(logits)

        pos_idx, neg_idx = _resolve_label_indices(model)
        prob_negative = probabilities[neg_idx]
        prob_positive = probabilities[pos_idx]
        final_score = prob_positive - prob_negative

        if final_score > 0.05:
            final_label = "positive"
        elif final_score < -0.05:
            final_label = "negative"
        else:
            final_label = "neutral"

        return (float(final_score), final_label)
    except Exception:
        logger.exception("sentiment_inference_failed text_len=%s", len(text))
        raise

