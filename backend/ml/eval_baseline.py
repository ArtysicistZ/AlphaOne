"""
Evaluate a base pre-trained model (no fine-tuning) on our test set.

This gives a sentence-level sentiment baseline to compare against
our entity-replacement fine-tuned model.

Usage:
    cd backend
    python -m ml.eval_baseline
    python -m ml.eval_baseline --model ProsusAI/finbert
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction

from ml.train_finbert import (
    load_labeled_data,
    compute_metrics,
    DEFAULT_MODEL,
    LABEL2ID,
    ID2LABEL,
)

MAX_LENGTH = 128
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Evaluate base model (no fine-tuning)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    print(f"=== Baseline: {args.model} (no fine-tuning) ===")
    print("This model does sentence-level sentiment — no entity conditioning.")
    print()

    # ── 1. Load data (same as training) ──────────────────────────────

    texts, subjects, labels = load_labeled_data()

    # ── 2. Same stratified split as training (seed=42) ───────────────
    #    Use original texts (no entity replacement) since base model
    #    doesn't know [TARGET]/[OTHER] tokens.
    #    Train 90% / Test 10% — same split as train_finbert.py.

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.1, random_state=SEED, stratify=labels,
    )

    print(f"Train: {len(train_texts)} (not used), Test: {len(test_texts)}")
    print()

    # ── 3. Load base model ───────────────────────────────────────────

    print(f"Loading base {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── 4. Run inference ─────────────────────────────────────────────

    def predict_batch(texts_list):
        all_logits = []
        batch_size = 64
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            all_logits.append(logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    print("Running inference on test set...")
    test_logits = predict_batch(test_texts)
    test_labels_np = np.array(test_labels)

    # ── 5. Compute metrics ───────────────────────────────────────────

    print("\n=== Test Results (Base Model) ===")
    test_metrics = compute_metrics(EvalPrediction(predictions=test_logits, label_ids=test_labels_np))
    for key in sorted(test_metrics):
        print(f"  {key}: {test_metrics[key]:.4f}")

    # ── 6. Show prediction distribution ──────────────────────────────

    test_preds = np.argmax(test_logits, axis=-1)

    print("\nBase model prediction distribution (test):")
    for label_id in sorted(ID2LABEL):
        count = int((test_preds == label_id).sum())
        total = len(test_preds)
        print(f"  {ID2LABEL[label_id]}: {count} ({100 * count / total:.1f}%)")

    print("\nActual label distribution (test):")
    for label_id in sorted(ID2LABEL):
        count = int((test_labels_np == label_id).sum())
        total = len(test_labels_np)
        print(f"  {ID2LABEL[label_id]}: {count} ({100 * count / total:.1f}%)")


if __name__ == "__main__":
    main()
