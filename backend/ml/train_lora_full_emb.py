"""
Ablation variant: LoRA + full embedding matrix trainable (~33M params).

This is the "0-buggy" configuration that accidentally achieved macro_f1=0.717
by making the entire embedding matrix trainable via modules_to_save.

The ONLY difference from train_finbert.py is in apply_lora():
  - modules_to_save=["classifier", "embed_tokens", "word_embeddings"]
    (trains full 38.6M embedding matrix)
  - No trainable_token_indices
  vs. train_finbert.py which uses trainable_token_indices for just 2 tokens.

Usage:
    cd backend
    python -m ml.train_finbert_0 --epochs 10
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

# Reuse shared components from main training script
from ml.train_base import (
    DEFAULT_MODEL,
    MAX_LENGTH,
    LABEL2ID,
    ID2LABEL,
    TARGET_TOKEN,
    OTHER_TOKEN,
    apply_entity_replacement,
    SentimentDataset,
    WeightedTrainer,
    load_labeled_data,
    compute_metrics,
    freeze_encoder_layers,
    init_special_token_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── LoRA (full embedding variant) ──────────────────────────────────────


def apply_lora_full_embed(model, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
    """Apply LoRA with full embedding matrix trainable (~33M trainable).

    This is the "buggy" variant that makes the entire embedding layer
    trainable via modules_to_save, resulting in ~33M trainable params
    instead of ~1.04M.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = ["query", "key", "value"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=["classifier", "embed_tokens", "word_embeddings"],
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        "LoRA+full_embed (rank=%d, alpha=%d) — trainable: %.3fM, frozen: %.1fM",
        rank, alpha, trainable / 1e6, frozen / 1e6,
    )
    model.print_trainable_parameters()
    return model


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: LoRA + full embedding (0-buggy config)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=str, default="./ml/models/finbert-sentiment"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Ablation: LoRA + Full Embedding (0-buggy) ===")
    print(f"Base model:       {args.model}")
    print(f"Method:           LoRA (rank={args.lora_rank}) + FULL embedding trainable")
    print(f"Epochs:           {args.epochs} (early stopping patience={args.patience})")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
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

    # ── 3. Stratified split: train 90% / test 10% ─────────────────

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        modified_texts, labels,
        test_size=0.1, random_state=args.seed, stratify=labels,
    )

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    print()

    # ── 4. Load tokenizer and add special tokens ─────────────────────

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [TARGET_TOKEN, OTHER_TOKEN],
    })
    logger.info("Added %d special tokens (vocab: %d)", num_added, len(tokenizer))

    # ── 5. Load model, resize embeddings, apply LoRA ─────────────────

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

    model = apply_lora_full_embed(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # ── 6. Tokenize (single segment — no sentence pairs) ─────────────

    print("Tokenizing...")
    train_enc = tokenizer(
        train_texts,
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt",
    )
    test_enc = tokenizer(
        test_texts,
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, torch.tensor(train_labels))
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
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Training...")
    print()
    train_result = trainer.train()
    print(f"\nTraining loss: {train_result.training_loss:.4f}")

    # ── 9. Evaluate on test set ───────────────────────────────────────

    print("\n=== Test Results ===")
    test_result = trainer.evaluate(test_dataset)
    for key in sorted(test_result):
        if key.startswith("eval_"):
            print(f"  {key}: {test_result[key]:.4f}")

    # ── 10. Save best model ──────────────────────────────────────────

    save_path = os.path.join(args.output_dir, "best")
    # Save LoRA adapters (small) + merge and save full model for inference
    lora_path = os.path.join(args.output_dir, "lora-adapters")
    model.save_pretrained(lora_path)
    print(f"LoRA adapters saved to: {lora_path}")

    # Merge LoRA weights into base model for simple inference loading
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Merged model saved to: {save_path}")


if __name__ == "__main__":
    main()
