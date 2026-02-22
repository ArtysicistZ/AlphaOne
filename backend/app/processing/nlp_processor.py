from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
import torch
import numpy as np
import os
import logging


_MODEL_ID = "ArtysicistZ/absa-deberta"
_MAX_LENGTH = 128
_TOKENIZER = None
_MODEL = None
logger = logging.getLogger(__name__)


def _get_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        logger.info("Loading sentiment model and tokenizer...")
        # DeBERTa-v3 fast tokenizer is broken — must use slow tokenizer
        # to correctly handle [TARGET]/[OTHER] special tokens
        _TOKENIZER = DebertaV2Tokenizer.from_pretrained(_MODEL_ID)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_sentiment(text: str) -> tuple[float, str]:
    if not text:
        return (0.0, "neutral")

    try:
        tokenizer, model = _get_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_LENGTH, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0].detach().cpu().numpy()
        probabilities = _softmax(logits)

        # {0: "bullish", 1: "bearish", 2: "neutral"}
        pred_idx = int(np.argmax(probabilities))
        label = model.config.id2label[pred_idx]
        score = float(probabilities[0] - probabilities[1])

        return (score, label)
    except Exception:
        logger.exception("sentiment_inference_failed text_len=%s", len(text))
        raise

def get_sentiment_batch(texts: list[str]) -> list[tuple[float, str]]:
    """Batch inference — single forward pass for multiple texts."""
    if not texts:
        return []

    tokenizer, model = _get_model()
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=_MAX_LENGTH, padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()       # shape: (N, 3)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # batch softmax

    results = []
    for probs in probabilities:
        pred_idx = int(np.argmax(probs))
        label = model.config.id2label[pred_idx]
        score = float(probs[0] - probs[1])
        results.append((score, label))
    return results