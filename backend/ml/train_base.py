"""
Fine-tune a pre-trained transformer for subject-conditioned sentiment classification.

Uses Entity Replacement approach (SEntFiN-style):
  1. In normalized text, replace the target subject ticker with "target"
  2. Replace all other known stock tickers with "other"
  3. Feed single-segment input: [CLS] modified_sentence [SEP]

Using plain vocabulary words ("target"/"other") instead of special tokens
([TARGET]/[OTHER]) means the model leverages pre-trained semantics from day one.
"target" ≈ "the subject of focus", "other" ≈ "something else" — no embedding
collapse risk.

Example:
  text = "AAPL is great but TSLA is doomed", subject = "AAPL"
  → input = "target is great but other is doomed"

Reference:
  Sinha et al. (2022) "SEntFiN 1.0: Entity-Aware Sentiment Analysis for
  Financial News" — achieved 94% accuracy with plain-word entity replacement.

Default base model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - RoBERTa-base further pre-trained on 58M tweets + fine-tuned for sentiment
  - Social media domain → closest match to Reddit financial text
  - BPE tokenizer handles slang/OOV better than FinBERT's WordPiece

Also supports: ProsusAI/finbert, microsoft/deberta-v3-base, or any
AutoModelForSequenceClassification-compatible model.

Fine-tuning methods:
  --lora           Use LoRA (Low-Rank Adaptation) — ~1.04M trainable params
  --freeze-layers  Use layer freezing — ~18-55M trainable params
  (default)        LoRA with rank=8 (recommended for <10K samples)

Prerequisites:
    pip install -r ml/requirements.txt

Usage:
    cd backend
    python -m ml.train_base                          # LoRA (default, recommended)
    python -m ml.train_base --no-lora                # layer freezing fallback
    python -m ml.train_base --lora-rank 16           # higher rank LoRA
    python -m ml.train_base --model ProsusAI/finbert
    python -m ml.train_base --no-lora --freeze-layers 0 --label-smoothing 0.15
"""

import argparse
import csv
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

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

from app.processing.sentiment_tagger.topic_definitions import SENTENCE_TOPIC_MAP
from app.processing.sentiment_tagger.tagger_logic import _GENERAL_TOPICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 128

# bullish=0, bearish=1, neutral=2
LABEL2ID = {"bullish": 0, "bearish": 1, "neutral": 2}
ID2LABEL = {0: "bullish", 1: "bearish", 2: "neutral"}

# Entity replacement tokens — plain vocabulary words (SEntFiN-style).
# Using existing words instead of special tokens means the model starts
# with meaningful pre-trained embeddings: "target" ≈ "the subject of focus",
# "other" ≈ "something else".  This avoids the embedding collapse problem
# where [TARGET] and [OTHER] initialized to the same mean vector never
# diverge (cosine similarity 0.999987 after training).
#
# Reference: Sinha et al. (2022) "SEntFiN 1.0: Entity-Aware Sentiment
# Analysis for Financial News" — achieved 94% accuracy with plain-word
# entity replacement.
TARGET_TOKEN = "target"
OTHER_TOKEN = "other"

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
    Replace the target subject ticker with "target" and all other known
    stock tickers with "other" in the normalized text.

    The normalized_text from the DB already has company names replaced with
    uppercase tickers (e.g., "apple" → "AAPL") by normalize_and_tag_sentence().

    Example:
        text="AAPL is great but TSLA is doomed", subject="AAPL"
        → "target is great but other is doomed"
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
    """Trainer with class-weighted CE loss + label smoothing + embedding divergence.

    Class weights address label imbalance (bullish ~13% vs neutral ~55%).
    Label smoothing prevents overconfidence in noisy LLM-generated labels.
    Divergence loss pushes "target" and "other" embeddings apart so the
    model learns to distinguish subject roles even better.

    Reference:
        Li et al. (2023) "Aspect-Pair Supervised Contrastive Learning for
        ABSA" — contrastive/divergence losses improve aspect distinction.
    """

    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0,
                 target_token_id: int = -1, other_token_id: int = -1,
                 divergence_weight: float = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.target_token_id = target_token_id
        self.other_token_id = other_token_id
        self.divergence_weight = divergence_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        cls_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )(logits, labels)

        # Divergence regularization: penalize high cosine similarity
        # between "target" and "other" embeddings
        if self.divergence_weight > 0 and self.target_token_id >= 0:
            # Handle both regular models and LoRA-wrapped models
            base = model.module if hasattr(model, "module") else model
            if hasattr(base, "get_input_embeddings"):
                emb_weight = base.get_input_embeddings().weight
            else:
                emb_weight = base.base_model.get_input_embeddings().weight

            target_emb = emb_weight[self.target_token_id]
            other_emb = emb_weight[self.other_token_id]

            cos_sim = nn.functional.cosine_similarity(
                target_emb.unsqueeze(0), other_emb.unsqueeze(0),
            )
            # Normalized to [0, 1]: 0 when orthogonal/opposite, 1 when identical
            div_loss = (1.0 + cos_sim) / 2.0

            loss = cls_loss + self.divergence_weight * div_loss.squeeze()
        else:
            loss = cls_loss

        return (loss, outputs) if return_outputs else loss


# ── Data Loading ─────────────────────────────────────────────────────────


DATA_DIR = Path(__file__).resolve().parent / "data" / "datasets"
AUDITED_CSV = DATA_DIR / "audit_working.csv"
SYNTHETIC_CSV = DATA_DIR / "synthetic_multitarget.csv"


def _load_csv(path: Path) -> tuple[list[str], list[str], list[int]]:
    """Load (text, subject, label) triples from a CSV file."""
    texts, subjects, labels = [], [], []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in LABEL2ID:
                skipped += 1
                continue
            texts.append(row["text"])
            subjects.append(row["subject"])
            labels.append(LABEL2ID[label])
    logger.info("Loaded %d rows from %s (%d skipped)", len(texts), path.name, skipped)
    return texts, subjects, labels


def load_labeled_data() -> tuple[list[str], list[str], list[int]]:
    """Load hand-audited labeled data from audit_working.csv."""
    return _load_csv(AUDITED_CSV)


def load_synthetic_multitarget() -> tuple[list[str], list[str], list[int]]:
    """Load synthetic multi-target sentences from synthetic_multitarget.csv."""
    return _load_csv(SYNTHETIC_CSV)


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

    # Balanced accuracy: average of per-class recalls (each class weighted equally)
    recalls = [metrics[f"recall_{name}"] for name in ID2LABEL.values()]
    metrics["balanced_accuracy"] = sum(recalls) / len(recalls)

    return metrics


# ── Layer Freezing ───────────────────────────────────────────────────────


def freeze_encoder_layers(model, num_layers: int):
    """Freeze the bottom N encoder layers of the transformer.

    Leaves embeddings trainable (needed for "target"/"other" fine-tuning).
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


# ── LoRA ─────────────────────────────────────────────────────────────


def apply_lora(model, tokenizer, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
    """Apply LoRA adapters to the model's attention layers.

    With rank=8: ~1.04M trainable params (vs ~35M for freeze-8, ~125M full).
    Every layer gets small low-rank adapters, preserving pre-trained weights.

    Trainable components:
      - LoRA adapters on Q/K/V attention (12 layers): ~442K params
      - Classifier head (dense 768×768 + out_proj 768×3): ~593K params
      - "target"/"other" token embeddings only: ~1.5K params
        (via trainable_token_indices — NOT the full 38.6M embedding matrix)

    References:
      - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
      - Frontiers (2025): LoRA outperforms full fine-tuning on <5K samples
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # Target the attention projection matrices in all layers.
    # These names work for both BERT (query/key/value) and RoBERTa.
    target_modules = ["query", "key", "value"]

    # Get token IDs for "target" and "other" so we can fine-tune ONLY those
    # embeddings instead of the full embedding matrix (38.6M).
    # These are existing vocabulary words — no need to add special tokens.
    # Requires peft >= 0.15.0.
    target_token_ids = tokenizer.encode(TARGET_TOKEN, add_special_tokens=False)
    other_token_ids = tokenizer.encode(OTHER_TOKEN, add_special_tokens=False)
    trainable_ids = target_token_ids + other_token_ids
    logger.info(
        "Trainable token indices: target=%s, other=%s",
        target_token_ids, other_token_ids,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=["classifier"],
        bias="none",
        trainable_token_indices=trainable_ids,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        "LoRA applied (rank=%d, alpha=%d) — trainable: %.3fM, frozen: %.1fM",
        rank, alpha, trainable / 1e6, frozen / 1e6,
    )
    model.print_trainable_parameters()
    return model


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer for subject-conditioned sentiment"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Max training epochs (default: 8, early stopping may end sooner)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5, tuned for full fine-tuning)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--lora", dest="lora", action="store_true", default=False,
                        help="Use LoRA fine-tuning (default: disabled)")
    parser.add_argument("--no-lora", dest="lora", action="store_false",
                        help="Full fine-tuning with layer freezing (default)")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8). Higher = more params, more capacity")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha scaling (default: 16, typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="LoRA dropout (default: 0.1)")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze bottom N encoder layers when --no-lora (0=none, default=0)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for noisy LLM labels (0=none, default=0.1)")
    parser.add_argument("--divergence-weight", type=float, default=0.1,
                        help="Weight for target/other embedding divergence loss (0=disabled, default=0.1)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience in epochs (0=disabled, default=2)")
    parser.add_argument(
        "--output-dir", type=str, default="./ml/models/deberta-v3-sentiment"
    )
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Disable synthetic multi-target data (default: enabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Sentiment Fine-Tuning (Entity Replacement) ===")
    print(f"Base model:       {args.model}")
    if args.lora:
        print(f"Method:           LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
    else:
        print(f"Method:           Full fine-tuning (freeze={args.freeze_layers} layers)")
    print(f"Epochs:           {args.epochs} (early stopping patience={args.patience})")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print(f"Label smoothing:  {args.label_smoothing}")
    print(f"Divergence wt:    {args.divergence_weight}")
    print(f"Device:           {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # ── 1. Load data ─────────────────────────────────────────────────

    texts, subjects, labels = load_labeled_data()
    if len(texts) == 0:
        print("ERROR: No labeled data found. Run llm_labeler first.")
        return

    print(f"Database:         {len(texts)} labeled pairs")

    # Merge synthetic multi-target sentences (unless --no-synthetic)
    if not args.no_synthetic:
        syn_texts, syn_subjects, syn_labels = load_synthetic_multitarget()
        texts.extend(syn_texts)
        subjects.extend(syn_subjects)
        labels.extend(syn_labels)
        print(f"Synthetic:        {len(syn_texts)} multi-target pairs")
        print(f"Combined:         {len(texts)} total pairs")
    print()

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

    # ── 3. Sentence-level stratified split: train 90% / test 10% ──
    #    Multi-target sentences produce multiple (text, subject) pairs.
    #    Splitting naively on pairs would leak the same sentence text into
    #    both train and test.  Instead, split on *unique sentences* first,
    #    then expand back to individual pairs.

    # Group indices by original sentence text (before entity replacement)
    sentence_groups: dict[str, list[int]] = defaultdict(list)
    for idx, txt in enumerate(texts):
        sentence_groups[txt].append(idx)

    unique_sentences = list(sentence_groups.keys())
    # Use the label of the first pair in each group for stratification
    unique_labels = [labels[sentence_groups[s][0]] for s in unique_sentences]

    train_sents, test_sents = train_test_split(
        unique_sentences,
        test_size=0.1, random_state=args.seed, stratify=unique_labels,
    )
    train_sent_set = set(train_sents)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for sent, indices in sentence_groups.items():
        for idx in indices:
            if sent in train_sent_set:
                train_texts.append(modified_texts[idx])
                train_labels.append(labels[idx])
            else:
                test_texts.append(modified_texts[idx])
                test_labels.append(labels[idx])

    print(f"Unique sentences: {len(unique_sentences)} "
          f"(train {len(train_sents)}, test {len(test_sents)})")
    print(f"Expanded pairs:   train {len(train_texts)}, test {len(test_texts)}")
    print()

    # ── 4. Load tokenizer ────────────────────────────────────────────
    #    No special tokens needed — "target" and "other" are plain vocabulary
    #    words with meaningful pre-trained embeddings (SEntFiN-style approach).

    print("Loading tokenizer...")
    # DeBERTa-v3 fast tokenizer conversion is broken; load slow tokenizer directly
    if "deberta" in args.model.lower():
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info("Tokenizer vocab size: %d", len(tokenizer))

    # ── 5. Load model, apply LoRA or freeze layers ────────────────────
    #    No resize_token_embeddings needed — vocabulary is unchanged.

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    # Update label mapping
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    if args.lora:
        model = apply_lora(
            model,
            tokenizer,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    elif args.freeze_layers > 0:
        freeze_encoder_layers(model, args.freeze_layers)

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

    # Get token IDs for divergence loss (plain vocabulary words)
    target_ids = tokenizer.encode(TARGET_TOKEN, add_special_tokens=False)
    other_ids = tokenizer.encode(OTHER_TOKEN, add_special_tokens=False)
    # Use first subword token for divergence loss (both are common single-token words)
    target_id = target_ids[0]
    other_id = other_ids[0]
    logger.info("Divergence loss token IDs: target=%d, other=%d", target_id, other_id)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        target_token_id=target_id,
        other_token_id=other_id,
        divergence_weight=args.divergence_weight,
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
    if args.lora:
        # Save LoRA adapters (small) + merge and save full model for inference
        lora_path = os.path.join(args.output_dir, "lora-adapters")
        model.save_pretrained(lora_path)
        print(f"LoRA adapters saved to: {lora_path}")

        # Merge LoRA weights into base model for simple inference loading
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to: {save_path}")
    else:
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
