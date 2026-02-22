"""
Error analysis: identify all misclassified samples from the fine-tuned model.

Loads the same data and applies the same sentence-level split as train_base.py
(seed=42), runs the best checkpoint on the test set, and prints every error
with the original text, subject, true label, predicted label, and confidence.

Usage:
    cd backend
    python -m ml.error_analysis
    python -m ml.error_analysis --model-dir ./ml/models/deberta-v3-sentiment/best
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ml.train_base import (
    load_labeled_data,
    load_synthetic_multitarget,
    load_error_targeted_synthetic,
    apply_entity_replacement,
    LABEL2ID,
    ID2LABEL,
    MAX_LENGTH,
)

SEED = 42
DEFAULT_MODEL_DIR = "./ml/models/deberta-v3-sentiment/best"


def main():
    parser = argparse.ArgumentParser(description="Error analysis on eval set")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Path to saved model (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV output path for errors")
    args = parser.parse_args()

    # ── 1. Load data (same as train_base.py) ───────────────────────────

    texts, subjects, labels = load_labeled_data()
    syn_texts, syn_subjects, syn_labels = load_synthetic_multitarget()
    texts.extend(syn_texts)
    subjects.extend(syn_subjects)
    labels.extend(syn_labels)
    err_texts, err_subjects, err_labels = load_error_targeted_synthetic()
    texts.extend(err_texts)
    subjects.extend(err_subjects)
    labels.extend(err_labels)
    print(f"Total pairs: {len(texts)}")

    # ── 2. Apply entity replacement ────────────────────────────────────

    modified_texts = [
        apply_entity_replacement(text, subject)
        for text, subject in zip(texts, subjects)
    ]

    # ── 3. Sentence-level split (same as train_base.py, seed=42) ──────

    sentence_groups: dict[str, list[int]] = defaultdict(list)
    for idx, txt in enumerate(texts):
        sentence_groups[txt].append(idx)

    unique_sentences = list(sentence_groups.keys())
    unique_labels = [labels[sentence_groups[s][0]] for s in unique_sentences]

    train_sents, test_sents = train_test_split(
        unique_sentences,
        test_size=0.1, random_state=SEED, stratify=unique_labels,
    )
    train_sent_set = set(train_sents)

    test_indices = []
    for sent, indices in sentence_groups.items():
        if sent not in train_sent_set:
            test_indices.extend(indices)

    test_texts_orig = [texts[i] for i in test_indices]
    test_subjects = [subjects[i] for i in test_indices]
    test_modified = [modified_texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"Test set: {len(test_indices)} pairs")

    # ── 4. Load model ──────────────────────────────────────────────────

    print(f"Loading model from {args.model_dir}...")
    if "deberta" in args.model_dir.lower():
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    # ── 5. Run inference ───────────────────────────────────────────────

    all_logits = []
    batch_size = 64
    for i in range(0, len(test_modified), batch_size):
        batch = test_modified[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits, dim=-1).numpy()
    all_preds = np.argmax(all_probs, axis=-1)
    test_labels_np = np.array(test_labels)

    # ── 6. Collect errors ──────────────────────────────────────────────

    errors = []
    for i in range(len(test_indices)):
        true_id = test_labels_np[i]
        pred_id = all_preds[i]
        if true_id != pred_id:
            errors.append({
                "text": test_texts_orig[i],
                "subject": test_subjects[i],
                "model_input": test_modified[i],
                "true_label": ID2LABEL[true_id],
                "pred_label": ID2LABEL[pred_id],
                "confidence": float(all_probs[i, pred_id]),
                "prob_bullish": float(all_probs[i, 0]),
                "prob_bearish": float(all_probs[i, 1]),
                "prob_neutral": float(all_probs[i, 2]),
            })

    # ── 7. Print summary ──────────────────────────────────────────────

    correct = int((all_preds == test_labels_np).sum())
    total = len(test_labels_np)
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"Errors:   {len(errors)}/{total}")

    # Confusion breakdown
    confusion = Counter()
    for e in errors:
        confusion[(e["true_label"], e["pred_label"])] += 1

    print("\nConfusion (true → pred):")
    for (true, pred), count in confusion.most_common():
        print(f"  {true:>8s} → {pred:<8s}: {count}")

    # ── 8. Print each error ───────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"ALL {len(errors)} ERRORS")
    print(f"{'='*80}")

    for i, e in enumerate(errors, 1):
        print(f"\n--- Error {i}/{len(errors)} ---")
        print(f"  Text:       {e['text'][:120]}")
        print(f"  Subject:    {e['subject']}")
        print(f"  Model sees: {e['model_input'][:120]}")
        print(f"  True:       {e['true_label']}")
        print(f"  Predicted:  {e['pred_label']} (conf={e['confidence']:.3f})")
        print(f"  Probs:      bull={e['prob_bullish']:.3f}  bear={e['prob_bearish']:.3f}  neut={e['prob_neutral']:.3f}")

    # ── 9. Optional CSV export ────────────────────────────────────────

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "text", "subject", "model_input", "true_label", "pred_label",
                "confidence", "prob_bullish", "prob_bearish", "prob_neutral",
            ])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\nErrors exported to {output_path}")


if __name__ == "__main__":
    main()
