"""
Fine-tune a pre-trained transformer for subject-conditioned sentiment classification.

Uses Entity Replacement approach (SEntFiN-style):
  1. In normalized text, replace the target subject ticker with [TARGET]
  2. Replace all other known stock tickers with [OTHER]
  3. Feed single-segment input: [CLS] modified_sentence [SEP]

Example:
  text = "AAPL is great but TSLA is doomed", subject = "AAPL"
  → input = "[TARGET] is great but [OTHER] is doomed"

Default base model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - RoBERTa-base further pre-trained on 58M tweets + fine-tuned for sentiment
  - Social media domain → closest match to Reddit financial text
  - BPE tokenizer handles slang/OOV better than FinBERT's WordPiece

Also supports: ProsusAI/finbert, microsoft/deberta-v3-base, or any
AutoModelForSequenceClassification-compatible model.

Prerequisites:
    pip install -r ml/requirements.txt

Usage:
    cd backend
    python -m ml.train_finbert
    python -m ml.train_finbert --model ProsusAI/finbert --freeze-layers 0
    python -m ml.train_finbert --freeze-layers 6 --label-smoothing 0.15 --patience 3
"""

import argparse
import logging
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)

from app.database.session import init_db, SessionLocal
from app.database.models import TrainingSentence, TrainingSentenceSubject
from app.processing.sentiment_tagger.topic_definitions import SENTENCE_TOPIC_MAP
from app.processing.sentiment_tagger.tagger_logic import _GENERAL_TOPICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH = 128

# bullish=0, bearish=1, neutral=2
LABEL2ID = {"bullish": 0, "bearish": 1, "neutral": 2}
ID2LABEL = {0: "bullish", 1: "bearish", 2: "neutral"}

# Entity replacement tokens
TARGET_TOKEN = "[TARGET]"
OTHER_TOKEN = "[OTHER]"

# All stock tickers (exclude general topics — MACRO/TECHNOLOGY keywords
# are NOT replaced in normalized text, so no entity replacement needed)
ALL_STOCK_TICKERS = frozenset(
    t for t in SENTENCE_TOPIC_MAP if t not in _GENERAL_TOPICS
)

# Pre-compiled regex: matches any stock ticker in one pass.
# Sorted by length descending so longer tickers match first.
_TICKER_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(t) for t in sorted(ALL_STOCK_TICKERS, key=len, reverse=True)
    ) + r")\b"
)


# ── Entity Replacement ──────────────────────────────────────────────────


def apply_entity_replacement(text: str, target_subject: str) -> str:
    """
    Replace the target subject ticker with [TARGET] and all other known
    stock tickers with [OTHER] in the normalized text.

    The normalized_text from the DB already has company names replaced with
    uppercase tickers (e.g., "apple" → "AAPL") by normalize_and_tag_sentence().

    Example:
        text="AAPL is great but TSLA is doomed", subject="AAPL"
        → "[TARGET] is great but [OTHER] is doomed"
    """
    def _replacer(match: re.Match) -> str:
        return TARGET_TOKEN if match.group() == target_subject else OTHER_TOKEN

    return _TICKER_PATTERN.sub(_replacer, text)


# ── Dataset ──────────────────────────────────────────────────────────────


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping pre-tokenized encodings + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Weighted Trainer ─────────────────────────────────────────────────────


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss + label smoothing.

    Class weights address label imbalance (bullish ~13% vs neutral ~55%).
    Label smoothing prevents overconfidence in noisy LLM-generated labels.
    """

    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Data Loading ─────────────────────────────────────────────────────────


def load_labeled_data() -> tuple[list[str], list[str], list[int]]:
    """Load labeled (sentence, subject, label) triples from the training DB."""
    init_db()
    db = SessionLocal()
    try:
        rows = (
            db.query(
                TrainingSentence.normalized_text,
                TrainingSentenceSubject.subject,
                TrainingSentenceSubject.sentiment_label,
            )
            .join(
                TrainingSentence,
                TrainingSentenceSubject.sentence_id == TrainingSentence.id,
            )
            .filter(TrainingSentenceSubject.sentiment_label.isnot(None))
            .all()
        )

        texts = []
        subjects = []
        labels = []
        skipped = 0
        for text, subject, label in rows:
            if label not in LABEL2ID:
                skipped += 1
                continue
            texts.append(text)
            subjects.append(subject)
            labels.append(LABEL2ID[label])

        logger.info("Loaded %d labeled pairs (%d skipped)", len(texts), skipped)
        return texts, subjects, labels
    finally:
        db.close()


# ── Metrics ──────────────────────────────────────────────────────────────


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Compute accuracy, per-class precision/recall/F1, and macro F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())

    metrics = {"accuracy": accuracy}
    f1_scores = []
    for label_id, label_name in ID2LABEL.items():
        tp = int(((predictions == label_id) & (labels == label_id)).sum())
        fp = int(((predictions == label_id) & (labels != label_id)).sum())
        fn = int(((predictions != label_id) & (labels == label_id)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[f"precision_{label_name}"] = precision
        metrics[f"recall_{label_name}"] = recall
        metrics[f"f1_{label_name}"] = f1
        f1_scores.append(f1)

    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    return metrics


# ── Layer Freezing ───────────────────────────────────────────────────────


def freeze_encoder_layers(model, num_layers: int):
    """Freeze the bottom N encoder layers of the transformer.

    Leaves embeddings trainable (needed for [TARGET]/[OTHER] special tokens).
    Leaves top encoder layers + classifier head trainable.
    Works with BERT, RoBERTa, DeBERTa, etc. via base_model_prefix.
    """
    base = getattr(model, model.base_model_prefix)
    encoder_layers = base.encoder.layer
    total_layers = len(encoder_layers)
    num_to_freeze = min(num_layers, total_layers)

    for i in range(num_to_freeze):
        for param in encoder_layers[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        "Froze %d/%d encoder layers — trainable: %.1fM, frozen: %.1fM",
        num_to_freeze, total_layers,
        trainable / 1e6, frozen / 1e6,
    )


# ── Special Token Init ──────────────────────────────────────────────────


def init_special_token_embeddings(model, num_new_tokens: int):
    """Initialize new special token embeddings to mean of existing embeddings.

    Better than random init — gives a neutral starting point instead of noise.
    Critical when training data is small (<5K samples).
    """
    if num_new_tokens == 0:
        return
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        existing_mean = embeddings.weight[:-num_new_tokens].mean(dim=0)
        for i in range(num_new_tokens):
            embeddings.weight[-(i + 1)] = existing_mean
    logger.info("Initialized %d special tokens to mean embedding", num_new_tokens)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer for subject-conditioned sentiment"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Max training epochs (default: 6, early stopping may end sooner)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--freeze-layers", type=int, default=8,
                        help="Freeze bottom N encoder layers (0=none, default=8)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for noisy LLM labels (0=none, default=0.1)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience in epochs (0=disabled, default=2)")
    parser.add_argument(
        "--output-dir", type=str, default="./ml/models/finbert-sentiment"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Sentiment Fine-Tuning (Entity Replacement) ===")
    print(f"Base model:       {args.model}")
    print(f"Epochs:           {args.epochs} (early stopping patience={args.patience})")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Freeze layers:    {args.freeze_layers}")
    print(f"Label smoothing:  {args.label_smoothing}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # ── 1. Load data ─────────────────────────────────────────────────

    texts, subjects, labels = load_labeled_data()
    if len(texts) == 0:
        print("ERROR: No labeled data found. Run llm_labeler first.")
        return

    # ── 2. Apply entity replacement ──────────────────────────────────

    modified_texts = [
        apply_entity_replacement(text, subject)
        for text, subject in zip(texts, subjects)
    ]

    print("Entity replacement examples:")
    for i in range(min(3, len(texts))):
        print(f"  Original: {texts[i]}")
        print(f"  Subject:  {subjects[i]}")
        print(f"  Modified: {modified_texts[i]}")
        print()

    dist = Counter(labels)
    print("Label distribution:")
    for label_id in sorted(dist):
        count = dist[label_id]
        print(f"  {ID2LABEL[label_id]}: {count} ({100 * count / len(labels):.1f}%)")
    print()

    # ── Compute class weights (inverse frequency) ────────────────────

    weights = compute_class_weight(
        "balanced",
        classes=np.array(sorted(LABEL2ID.values())),
        y=np.array(labels),
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print("Class weights (inverse frequency):")
    for label_id in sorted(ID2LABEL):
        print(f"  {ID2LABEL[label_id]}: {weights[label_id]:.3f}")
    print()

    # ── 3. Stratified split: train 80% / val 10% / test 10% ─────────

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        modified_texts, labels,
        test_size=0.1, random_state=args.seed, stratify=labels,
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=0.111, random_state=args.seed, stratify=train_labels,
        # 0.111 of 90% ≈ 10% of total
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    print()

    # ── 4. Load tokenizer and add special tokens ─────────────────────

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [TARGET_TOKEN, OTHER_TOKEN],
    })
    logger.info("Added %d special tokens (vocab: %d)", num_added, len(tokenizer))

    # ── 5. Load model, resize embeddings, freeze layers ──────────────

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize [TARGET]/[OTHER] to mean embedding instead of random noise
    init_special_token_embeddings(model, num_added)

    # Update label mapping
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    # Freeze bottom encoder layers to prevent overfitting on small data
    if args.freeze_layers > 0:
        freeze_encoder_layers(model, args.freeze_layers)

    # ── 6. Tokenize (single segment — no sentence pairs) ─────────────

    print("Tokenizing...")
    train_enc = tokenizer(
        train_texts,
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt",
    )
    test_enc = tokenizer(
        test_texts,
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, torch.tensor(train_labels))
    val_dataset = SentimentDataset(val_enc, torch.tensor(val_labels))
    test_dataset = SentimentDataset(test_enc, torch.tensor(test_labels))

    # ── 7. Training arguments ────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=args.seed,
    )

    # ── 8. Train ─────────────────────────────────────────────────────

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Training...")
    print()
    train_result = trainer.train()
    print(f"\nTraining loss: {train_result.training_loss:.4f}")

    # ── 9. Evaluate ──────────────────────────────────────────────────

    print("\n=== Validation Results ===")
    val_result = trainer.evaluate()
    for key in sorted(val_result):
        if key.startswith("eval_"):
            print(f"  {key}: {val_result[key]:.4f}")

    print("\n=== Test Results ===")
    test_result = trainer.evaluate(test_dataset)
    for key in sorted(test_result):
        if key.startswith("eval_"):
            print(f"  {key}: {test_result[key]:.4f}")

    # ── 10. Save best model ──────────────────────────────────────────

    save_path = os.path.join(args.output_dir, "best")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
