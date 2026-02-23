from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
import torch
import numpy as np
import os
import logging


_MODEL_ID = "ArtysicistZ/deberta-absa-v2"
_MAX_LENGTH = 128
_TOKENIZER = None
_MODEL = None
logger = logging.getLogger(__name__)


def _get_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        logger.info("Loading sentiment model and tokenizer...")
        # DeBERTa-v3 fast tokenizer is broken — must use slow tokenizer
        # to correctly handle "target"/"other" entity replacement tokens
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


def get_sentiment_batch_with_attention(
    texts: list[str],
) -> list[tuple[float, str, list[list[float]], list[str]]]:
    """Batch inference with last-layer attention extraction.

    Returns list of (score, label, attention_matrix, tokens).
      attention_matrix: (real_len x real_len) head-averaged from last layer.
      tokens: subword token strings for axis labels.
    """
    if not texts:
        return []

    tokenizer, model = _get_model()
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=_MAX_LENGTH, padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits.detach().cpu().numpy()
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # Last layer attention: (N, 12_heads, seq, seq) → avg heads → (N, seq, seq)
    last_layer_attn = outputs.attentions[-1].mean(dim=1).detach().cpu().numpy()
    attention_mask = inputs["attention_mask"].numpy()
    input_ids = inputs["input_ids"].numpy()

    results = []
    for i, probs in enumerate(probabilities):
        pred_idx = int(np.argmax(probs))
        label = model.config.id2label[pred_idx]
        score = float(probs[0] - probs[1])

        # Trim to actual (non-padding) tokens
        real_len = int(attention_mask[i].sum())
        trimmed = last_layer_attn[i, :real_len, :real_len]

        # Row-normalize after trimming padding columns
        row_sums = trimmed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        normalized = (trimmed / row_sums).tolist()

        # Decode token strings for axis labels
        token_ids = input_ids[i, :real_len].tolist()
        tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]

        results.append((score, label, normalized, tokens))

    return results