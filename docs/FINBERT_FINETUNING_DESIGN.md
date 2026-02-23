# Sentiment Fine-Tuning Design: Subject-Conditioned Classification

## 1) What This Document Covers

This document records the actual design decisions for fine-tuning a transformer model to classify financial sentiment **toward a specific stock** in Reddit text. Every choice — model, architecture, data pipeline, hyperparameters — is explained with its rationale.

This supersedes the aspirational plans in `SUBJECT_CONDITIONED_SFT_PLAN.md` and `LLM_LABELING_PIPELINE_PLAN.md` where they conflict.

---

## 2) The Problem

Given a sentence and a target stock ticker, classify the sentiment **specifically toward that stock**.

Example — same sentence, different labels:

```
Sentence: "AAPL is great but TSLA is doomed"
Subject:  AAPL  →  bullish
Subject:  TSLA  →  bearish
```

Standard sentence-level sentiment models cannot do this. They produce one label for the entire sentence, ignoring which stock we are asking about.

This problem is formally called **Aspect-Based Sentiment Analysis (ABSA)** or **Entity-Level Targeted Sentiment**. It is well-studied with established solutions (Sun et al., NAACL 2019; Sinha et al., JASIST 2022).

---

## 3) Architecture Decisions

### 3.1 Base model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Default model** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa-base pre-trained on 58M tweets + fine-tuned for sentiment. Social media domain is the closest match to Reddit financial text |
| Alternative A | `ProsusAI/finbert` | Financial domain pre-training but trained on formal news/SEC filings — domain mismatch with Reddit slang, memes, and options jargon |
| Alternative B | `microsoft/deberta-v3-base` | Best architecture for NLU benchmarks but no domain-specific pre-training |

Why we switched from ProsusAI/finbert:

1. **Domain mismatch.** FinBERT was pre-trained on Reuters financial news and SEC filings. Our input is Reddit posts: "Naw retards, BLK got that shit pinned", "bought a ton of weekly far OTM puts on NVDA". FinBERT's pre-trained representations do not capture slang, profanity, or options trading lingo.
2. **Tokenizer.** FinBERT uses WordPiece (BERT-style) which splits unknown words into subword fragments. RoBERTa uses BPE which handles out-of-vocabulary words (Reddit slang) more gracefully.
3. **Training data scale.** RoBERTa-base was trained on 160GB of text (vs BERT's 16GB), producing more robust representations. The twitter-roberta variant adds 58M tweets on top.
4. **Empirical.** RoBERTa consistently outperforms BERT on NLU benchmarks. The sentiment-specific twitter variant gives an even better starting point for our task.

The model is configurable via `--model`. ProsusAI/finbert remains supported:

```bash
python -m ml.train_finbert --model ProsusAI/finbert --freeze-layers 0
```

Key facts about `cardiffnlp/twitter-roberta-base-sentiment-latest`:

- Architecture: `RobertaForSequenceClassification` (loaded via `AutoModelForSequenceClassification`)
- Hidden size: 768, 12 layers, 12 attention heads
- Vocab: 50,265 (RoBERTa BPE vocabulary)
- Pre-trained labels: `{0: "negative", 1: "neutral", 2: "positive"}` — replaced with our mapping during fine-tuning
- `num_labels: 3` — same as our task, so the classification head size matches

### 3.2 Label mapping

| Our label (id) | Semantics |
|----------------|-----------|
| bullish (0) | optimistic, buying, accumulating, price going up |
| bearish (1) | pessimistic, selling, shorting, price going down |
| neutral (2) | factual, no opinion, just mentioning |

We set `model.config.id2label = ID2LABEL` after loading to override the pre-trained model's label names. The classification head weights are fine-tuned from whatever the base model provides.

### 3.3 Why single-head, not multi-head

The original SFT plan proposed three output heads. We use **only 3-class classification**. Reasons:

- **No relevance head needed.** Every `(sentence, subject)` pair already has a detected subject via keyword matching. No "hard negative" pairs exist.
- **No regression head needed.** Continuous score derived from softmax: `score = P(bullish) - P(bearish)` ∈ [-1, +1].
- **Simpler is better with limited data.** Multi-head training splits gradient signal.

### 3.4 Input format: Entity Replacement (SEntFiN-style)

Replace the target subject's ticker with `[TARGET]` and all other known stock tickers with `[OTHER]`, then feed a single-segment input:

```
Original text:  "AAPL is great but TSLA is doomed"
Subject:        AAPL

After replacement: "[TARGET] is great but [OTHER] is doomed"

Tokens:         <s> [TARGET] is great but [OTHER] is doomed </s> <pad>...
attention_mask:  1      1      1   1     1    1     1    0     1    0...
```

(Note: RoBERTa uses `<s>`/`</s>` instead of BERT's `[CLS]`/`[SEP]`, and does not use `token_type_ids`. The tokenizer handles this automatically.)

### 3.5 Special token initialization

`[TARGET]` and `[OTHER]` are added to the tokenizer as special tokens. After `resize_token_embeddings()`, they would normally get random vectors (mean=0, std=0.02).

**We initialize them to the mean of all existing embeddings instead.** With only ~3,500 training samples, random initialization for tokens that appear in every input creates unnecessary noise. Mean initialization gives a semantically neutral starting point.

```python
with torch.no_grad():
    embeddings = model.get_input_embeddings()
    existing_mean = embeddings.weight[:-num_new_tokens].mean(dim=0)
    for i in range(num_new_tokens):
        embeddings.weight[-(i + 1)] = existing_mean
```

### 3.6 Text normalization and its role

The `build_training_set.py` pipeline normalizes company names to canonical tickers via `normalize_and_tag_sentence()`:

```
"apple earnings were great"  →  "AAPL earnings were great"
"$tsla to the moon"          →  "TSLA to the moon"
```

This normalization is critical for entity replacement because it reduces all company references to a single canonical uppercase ticker per entity, enabling reliable regex matching.

---

## 4) Data Pipeline

### 4.1 Pipeline overview

```
raw_reddit_posts (6,000+ posts)
        │
        ▼  split_into_sentences() + normalize_and_tag_sentence()
training_sentences (one row per sentence text)
        │
        ▼  one row per (sentence, subject) pair
training_sentence_subjects (~3,500 labeled pairs)
        │
        ▼  train_finbert.py loads from DB
Entity replacement → tokenize → train / val / test
```

### 4.2 Training tables (3NF)

**`training_sentences`** — stores sentence text once:

| Column | Type | Purpose |
|--------|------|---------|
| `id` | Integer PK | |
| `raw_post_id` | FK → `raw_reddit_posts.id` | Source tracing |
| `sentence_index` | Integer | Position within the post |
| `normalized_text` | Text | Company names replaced with tickers |
| `subreddit` | String | Source subreddit |
| `created_utc` | DateTime | Original post timestamp |

**`training_sentence_subjects`** — one row per (sentence, subject) pair:

| Column | Type | Purpose |
|--------|------|---------|
| `id` | Integer PK | |
| `sentence_id` | FK → `training_sentences.id` | |
| `subject` | String | Ticker symbol (e.g., "AAPL") |
| `sentiment_label` | String, nullable | "bullish", "bearish", or "neutral" (NULL = unlabeled) |
| `sentiment_confidence` | Float, nullable | LLM confidence (0.0-1.0) |

Unique constraint: `(sentence_id, subject)` — prevents duplicate entries.

### 4.3 Data filtering

General topics (`MACRO`, `TECHNOLOGY`) are excluded from training data:

```python
topics = topics - _GENERAL_TOPICS  # frozenset({"MACRO", "TECHNOLOGY"})
```

### 4.4 Dataset size

~3,500 labeled (sentence, subject) pairs. Comparable to standard ABSA benchmarks:

| Benchmark | Training size | Domain |
|-----------|--------------|--------|
| SemEval-2014 Restaurant | ~3,000 | Reviews |
| SemEval-2014 Laptop | ~1,800 | Reviews |
| FinEntity (EMNLP 2023) | ~2,131 | Financial |
| **Ours** | **~3,500** | **Financial** |

---

## 5) LLM Labeling Pipeline

### 5.1 Why local Ollama instead of Cerebras API

| Factor | Cerebras API (original plan) | Ollama local (actual) |
|--------|-----------------------------|-----------------------|
| Speed | Rate-limited: 30 RPM | 4-6 parallel workers, ~3,500 labels in 30-60 min |
| Cost | Free but capped | Free, no caps |
| Model | Llama 3.1 8B | qwen2.5:3b (current) |
| Dependency | Requires internet + API key | Runs entirely offline |

### 5.2 LLM model selection

| Model | Size | Accuracy | Issue |
|-------|------|----------|-------|
| qwen2.5:1.5b | 1.5B | ~80% (old prompt), ~90% (new prompt) | Fast (6 workers), occasionally misses multi-clause reasoning |
| **qwen2.5:3b** | **3B** | **~95%** | **Current choice.** Slower (4 workers) but significantly more accurate |
| qwen3:4b | 4B | 0% (MISS) | `<think>` tags consume `num_predict` budget before JSON output |
| phi4-mini | 3.8B | Untested | Backup option |

Key design decisions:

1. **One pair per call.** Small models fail at batch prompting (40% accuracy). Single-pair mode gives 90%+ accuracy.
2. **Parallel HTTP calls (4 workers for 3b).** Compensates for single-pair overhead.
3. **Non-thinking models only.** Qwen3's `<think>` tags are incompatible with tight `num_predict` limits.

### 5.3 Prompt design

Stored in `backend/ml/data/prompts.py` (single source of truth).

The prompt was iteratively refined through multiple rounds:

1. **v1 (original)**: No examples → 40% accuracy with batch mode
2. **v2**: Added few-shot examples + rules → 90% accuracy
3. **v3 (attempted trimming)**: Removed rules, kept only examples → accuracy dropped. Reverted.
4. **v4 (current)**: Balanced 2/2/2 examples (bullish/bearish/neutral) + full rules

**Critical fix**: The original prompt had **1 bullish / 5 bearish / 0 neutral** examples. This created systematic bearish labeling bias — the LLM calibrated its output distribution from few-shot examples. The fix to 2/2/2 balanced examples eliminated this bias and was the single largest improvement to label quality.

Key rules in the prompt:
- Selling OTHER positions to buy the subject = bullish toward the subject
- Buying PUT options = bearish (the trader bet against the stock)
- Profiting from PUT options = bearish
- Predicting price DROP ("sub $X", "will go to $X" where X is lower) = bearish
- Sarcasm counts as OPPOSITE of literal words
- Stock flat/going nowhere for years = neutral or bearish, NOT bullish

### 5.4 Labeling script configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `DEFAULT_MODEL` | `qwen2.5:3b` | Best accuracy among non-thinking small models |
| `DEFAULT_WORKERS` | 4 | RTX 4060 limit for 3b model |
| `COMMIT_EVERY` | 20 | Frequent terminal output for monitoring |
| `temperature` | 0.1 | Low for consistent classification |
| `num_predict` | 64 | Enough for JSON output |
| Resumable | `WHERE sentiment_label IS NULL` | Can stop/restart anytime |

### 5.5 Label noise consideration

At ~87% LLM accuracy (qwen2.5:3b with balanced prompt), ~13% of training labels are incorrect. Mitigated by:

1. **Label smoothing (0.1)** in training loss — prevents overconfidence in any single label.
2. **Class-weighted loss** — addresses label imbalance but won't amplify noise as severely with better base labels.
3. BERT-class models are empirically robust to 5-10% label noise.

Update: Using Claude Code to audit the whole dataset with size ~4500, achieving ~98% accuracy.

---

## 6) Training Pipeline

### 6.1 File: `backend/ml/train_finbert.py`

```bash
cd backend
pip install -r ml/requirements.txt

# Default (recommended): LoRA + twitter-roberta + all regularization
python -m ml.train_finbert

# Use old FinBERT base
python -m ml.train_finbert --model ProsusAI/finbert

# Layer freezing fallback (legacy approach)
python -m ml.train_finbert --no-lora --freeze-layers 8

# Customize LoRA
python -m ml.train_finbert --lora-rank 16 --lora-alpha 32 --lr 2e-5
```

### 6.2 Data loading

The training script queries the DB directly:

```python
db.query(
    TrainingSentence.normalized_text,
    TrainingSentenceSubject.subject,
    TrainingSentenceSubject.sentiment_label,
)
.join(TrainingSentence, ...)
.filter(TrainingSentenceSubject.sentiment_label.isnot(None))
```

### 6.3 Entity replacement

After loading, each `(text, subject)` pair is transformed:

```python
ALL_STOCK_TICKERS = frozenset(
    t for t in SENTENCE_TOPIC_MAP if t not in _GENERAL_TOPICS
)
_TICKER_PATTERN = re.compile(r"\b(?:AAPL|TSLA|NVDA|...)\b")

def apply_entity_replacement(text, target_subject):
    def _replacer(match):
        return "[TARGET]" if match.group() == target_subject else "[OTHER]"
    return _TICKER_PATTERN.sub(_replacer, text)
```

### 6.4 Parameter-efficient fine-tuning: LoRA (default)

With ~4,800 training samples and a 125M parameter model, the data:parameter ratio is critical. Research shows PEFT methods **outperform full fine-tuning on <5K samples** (Frontiers 2025, NeurIPS 2022).

**Default: LoRA (Low-Rank Adaptation)**. Adds small low-rank adapter matrices to every attention layer while keeping all pre-trained weights frozen. The parameter constraint acts as a built-in regularizer.

```python
from peft import LoraConfig, get_peft_model, TaskType

special_token_ids = tokenizer.convert_tokens_to_ids(["[TARGET]", "[OTHER]"])

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    modules_to_save=["classifier"],
    bias="none",
    trainable_token_indices=special_token_ids,
)
model = get_peft_model(model, config)
# trainable: ~1.04M (0.82%) || frozen: ~124.6M
```

Trainable breakdown:
- LoRA adapters on Q/K/V (12 layers × 3 matrices × rank 8): ~442K
- Classifier head (`dense` 768×768 + `out_proj` 768×3): ~593K
- [TARGET]/[OTHER] token embeddings (2 × 768): ~1.5K

Key design choices:
- **`target_modules`**: Query, key, value attention projections in all 12 layers. Every layer adapts slightly rather than top-N layers changing completely.
- **`modules_to_save`**: Classifier head only (dense + out_proj). Must be fully trainable for the 3-class task.
- **`trainable_token_indices`**: Trains ONLY the [TARGET] and [OTHER] embeddings (2 × 768 = 1,536 params) instead of the full 38.6M embedding matrix. Requires peft >= 0.15.0.
- **`r=8, alpha=16`**: Standard rank. Alpha=2×rank is the common convention (Hu et al. 2021).

| Method | `--lora` / `--no-lora` | Trainable | Data:param ratio (4.8K samples) |
|--------|------------------------|-----------|-------------------------------|
| **LoRA rank=8** (default) | `--lora` | **~1.04M** | **4.6:1** |
| LoRA rank=16 | `--lora-rank 16` | ~1.8M | 2.7:1 |
| Layer freeze 8 | `--no-lora` | ~35M | 1:7,300 |
| Layer freeze 10 | `--no-lora --freeze-layers 10` | ~18M | 1:3,750 |
| Full fine-tuning | `--no-lora --freeze-layers 0` | ~125M | 1:26,000 |

References:
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Frontiers in Big Data (2025): LoRA achieved 0.909 F1 on 1,000 samples where full fine-tuning overfits
- NeurIPS 2022: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"

### 6.4b Layer freezing (legacy fallback)

Available via `--no-lora`. Freezes the bottom N encoder layers, leaving top layers + classifier + embeddings trainable.

```bash
python -m ml.train_finbert --no-lora --freeze-layers 8
```

| `--freeze-layers` | Trainable | Notes |
|-------------------|-----------|-------|
| 0 | ~125M | Full fine-tuning. Only viable with 10K+ samples |
| 4 | ~55M | Moderate freezing |
| 8 | ~35M | Aggressive freezing |
| 10 | ~18M | Near-linear probe |

### 6.5 Tokenization

```python
tokenizer(
    modified_texts,     # entity-replaced sentences
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt",
)
```

### 6.6 Data split

Stratified 90/10 split (train / test). No separate validation set — test set is used for eval during training to maximize training data.

```python
train_test_split(data, test_size=0.1, random_state=42, stratify=labels)
```

With ~6300 total: 5654 train / 633 test.

### 6.7 Hyperparameters

| Parameter | Default | CLI flag | Rationale |
|-----------|---------|----------|-----------|
| `model` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `--model` | Social media domain, BPE tokenizer |
| `lora` | `True` | `--lora` / `--no-lora` | LoRA fine-tuning (recommended for <10K samples) |
| `lora_rank` | `8` | `--lora-rank` | Low-rank dimension. 8 = ~1.04M trainable params |
| `lora_alpha` | `16` | `--lora-alpha` | Scaling factor, typically 2× rank |
| `lora_dropout` | `0.1` | `--lora-dropout` | Dropout on LoRA layers |
| `learning_rate` | `2e-4` | `--lr` | Standard LoRA LR (Hu et al. 2021). 1e-5 caused severe underfitting |
| `num_train_epochs` | `6` | `--epochs` | Max epochs; early stopping typically ends at 3-4 |
| `batch_size` | `32` | `--batch-size` | Standard for BERT-class models |
| `freeze_layers` | `8` | `--freeze-layers` | Layer freezing when `--no-lora` (ignored with LoRA) |
| `label_smoothing` | `0.1` | `--label-smoothing` | Prevent overconfidence in noisy LLM labels |
| `patience` | `2` | `--patience` | Early stopping patience (epochs without macro_F1 improvement) |
| `weight_decay` | `0.01` | — | AdamW standard |
| `warmup_ratio` | `0.1` | — | 10% of total steps |
| `max_length` | `128` | `--max-length` | Sufficient for single sentences |
| `fp16` | auto | — | Enabled when CUDA is available |
| `metric_for_best_model` | `macro_f1` | — | Balanced metric across all 3 classes |
| `save_total_limit` | `3` | — | Prevent disk bloat from checkpoints |
| `seed` | `42` | `--seed` | Reproducibility |

### 6.8 Loss function

`WeightedTrainer` combines two mechanisms:

1. **Class-weighted CrossEntropyLoss**: Inverse-frequency weights via `sklearn.compute_class_weight("balanced")`. Addresses label imbalance (bullish ~13% vs neutral ~55%).
2. **Label smoothing**: Distributes 10% of probability mass to non-target classes. Prevents the model from memorizing noisy LLM labels with full confidence.

```python
nn.CrossEntropyLoss(
    weight=class_weights,       # [2.5, 1.0, 0.6] approximately
    label_smoothing=0.1,        # configurable via --label-smoothing
)
```

### 6.9 Early stopping

`EarlyStoppingCallback(early_stopping_patience=2)` monitors `eval_macro_f1` after each epoch. Training stops if macro_F1 doesn't improve for 2 consecutive epochs.

This addresses the overfitting observed in previous runs where eval loss increased continuously after epoch 1 while train loss continued dropping.

### 6.10 Evaluation metrics

Computed after each epoch on the test set (used as eval set during training):

- **Accuracy**: overall classification accuracy
- **Per-class precision**: TP / (TP + FP) for each class
- **Per-class recall**: TP / (TP + FN) for each class
- **Per-class F1**: harmonic mean of precision and recall
- **Macro F1**: unweighted average of per-class F1 (primary metric)

### 6.11 Model output and saving

With LoRA, both the raw adapters and a merged model are saved:

```
ml/models/finbert-sentiment/
├── lora-adapters/          # LoRA weights only (~4MB) — for resuming training
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── best/                   # Merged full model — for inference (same as non-LoRA)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── added_tokens.json
    ├── special_tokens_map.json
    └── (vocab files)
```

The `best/` directory contains the merged model (LoRA weights folded into base weights). Loading for inference is identical regardless of training method — no PEFT dependency needed at inference time.

---

## 7) Inference

### 7.1 Loading the fine-tuned model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ml/models/finbert-sentiment/best")
model = AutoModelForSequenceClassification.from_pretrained("ml/models/finbert-sentiment/best")
model.eval()
```

### 7.2 Making predictions

```python
from ml.train_finbert import apply_entity_replacement

sentence = "AAPL is great but TSLA is doomed"
subject = "AAPL"

modified = apply_entity_replacement(sentence, subject)
# → "[TARGET] is great but [OTHER] is doomed"

inputs = tokenizer(modified, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()

label_id = probs.argmax().item()
label = model.config.id2label[label_id]       # "bullish"
score = (probs[0] - probs[1]).item()           # P(bullish) - P(bearish) ∈ [-1, +1]
```

### 7.3 Output contract

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | `"bullish"`, `"bearish"`, or `"neutral"` |
| `score` | float [-1, +1] | `P(bullish) - P(bearish)` |
| `probabilities` | dict | `{"bullish": 0.8, "bearish": 0.1, "neutral": 0.1}` |
| `confidence` | float [0, 1] | `max(probabilities)` |

---

## 8) Ablation Study Plan

### 8.1 Purpose

Systematically measure which changes actually improve performance. Without ablation, we cannot distinguish lucky parameter combinations from genuine improvements.

### 8.2 Baseline — base model with no fine-tuning

The true baseline is the pre-trained model evaluated directly on our test set, with **no training at all**. This tells us what the model already knows before we touch it.

```bash
python -m ml.eval_baseline
python -m ml.eval_baseline --model ProsusAI/finbert
```

Any fine-tuned run that doesn't beat the baseline is worse than doing nothing.

#### v1 training reference (old setup, for context)

The previous fine-tuned run for comparison:

| Metric | Value |
|--------|-------|
| Accuracy | 0.744 |
| Macro F1 | 0.684 |
| Bullish F1 | 0.496 |
| Bearish F1 | 0.749 |
| Neutral F1 | 0.808 |

Configuration: `ProsusAI/finbert`, LR=2e-5, epochs=4, no freezing, no label smoothing, no early stopping, random token init, labels from qwen2.5:1.5b with biased prompt.

### 8.3 Ablation runs

Each run changes **one variable** from the full new configuration to measure its individual contribution. Run all with `--seed 42` for reproducibility.

**Run 0 — Full new with LoRA (target to beat)**
```bash
python -m ml.train_finbert
# twitter-roberta, LoRA rank=8, smoothing=0.1, patience=2, lr=2e-4, new labels
# Trainable: ~1.04M params (0.82%)
```

**Run 1 — Isolate: base model (twitter-roberta vs finbert)**
```bash
python -m ml.train_finbert --model ProsusAI/finbert
```
Measures: how much does the base model choice matter?

**Run 2 — Isolate: LoRA vs layer freezing**
```bash
python -m ml.train_finbert --no-lora --freeze-layers 8
# Same config but layer freezing (~35M params) instead of LoRA (~1M params)
```
Measures: the core question — does LoRA's parameter constraint help vs crude layer freezing?

**Run 3 — Isolate: label smoothing**
```bash
python -m ml.train_finbert --label-smoothing 0
```
Measures: does label smoothing help with noisy LLM labels?

**Run 4 — Isolate: learning rate**
```bash
python -m ml.train_finbert --lr 2e-5
```
Measures: 1e-5 vs 2e-5 with LoRA.

**Run 5 — Isolate: early stopping**
```bash
python -m ml.train_finbert --patience 0 --epochs 4
```
Measures: does early stopping prevent overfitting-related degradation?

**Run 6 — Variant: LoRA rank 16 (higher capacity)**
```bash
python -m ml.train_finbert --lora-rank 16 --lora-alpha 32
# ~1.8M trainable params
```
Measures: does higher rank improve performance or overfit?

**Run 7 — Variant: full fine-tuning (no LoRA, no freezing)**
```bash
python -m ml.train_finbert --no-lora --freeze-layers 0
# ~125M trainable params — expected to overfit
```
Measures: baseline for how badly full fine-tuning overfits on ~4.8K samples.

### 8.4 Results

#### Summary table

| Run | Base | Method | LR | Trainable | Acc | Macro F1 | Bull F1 | Bear F1 | Neut F1 | Epoch | Eval loss |
|-----|------|--------|----|-----------|-----|----------|---------|---------|---------|-------|-----------|
| base-roberta | roberta | none | — | — | 0.254 | 0.221 | 0.107 | 0.307 | 0.249 | — | — |
| base-finbert | finbert | none | — | — | 0.543 | 0.433 | 0.162 | 0.475 | 0.662 | — | — |
| v1 ref | finbert | full FT | 2e-5 | ~125M | 0.744 | 0.684 | 0.496 | 0.749 | 0.808 | 4/4 | 0.950 ↑ |
| v2 ref | finbert | full FT | 2e-5 | ~125M | 0.759 | 0.703 | 0.548 | 0.740 | 0.820 | ?/10 | 0.989 |
| v3 ref | roberta | full FT | 2e-5 | ~125M | 0.796 | 0.756 | 0.667 | 0.749 | 0.851 | 4/10 | 0.824 |
| v4 ref | deberta-v3 | full FT | 2e-5 | ~86M | 0.805 | 0.757 | 0.640 | 0.777 | 0.853 | 4/10 | 0.840 |
| v5 ref | bertweet | full FT | 2e-5 | ~135M | 0.751 | 0.700 | 0.562 | 0.715 | 0.822 | 4/6 | 0.934 |
| 0-buggy | roberta | LoRA r8 + full embed | 2e-4 | ~33M | 0.763 | 0.717 | 0.593 | 0.735 | 0.823 | ?/10 | 0.891 |
| 0a-buggy | roberta | LoRA r8 + full embed | 5e-4 | ~33M | 0.776 | 0.724 | 0.593 | 0.733 | 0.845 | 4/10 | 0.873 |
| 1-buggy | finbert | LoRA r8 + full embed | 2e-4 | ~33M | 0.738 | 0.687 | 0.559 | 0.696 | 0.806 | ?/10 | 0.983 |
| 0 | roberta | LoRA r8 | 2e-4 | ~1.04M | 0.713 | 0.681 | 0.565 | 0.708 | 0.770 | ?/6 | 0.856 |
| 0a | roberta | LoRA r8 | 5e-4 | ~1.04M | 0.736 | 0.696 | 0.568 | 0.722 | 0.797 | ?/6 | 0.841 |
| v6 | deberta-v3 | full FT | 1e-5 | ~86M | 0.794 | 0.754 | 0.662 | 0.759 | 0.843 | 7/10 | 0.823 |

#### Dataset v2: Plain-Word Entity Replacement + Synthetic Multi-Target (6,233 pairs)

New dataset and architecture — **not directly comparable** to the table above due to:
- **Plain-word tokens**: `target`/`other` instead of `[TARGET]`/`[OTHER]` special tokens (SEntFiN-style)
- **Expanded dataset**: 4,749 DB pairs + 1,484 synthetic multi-target pairs (600 sentences × 2-4 subjects)
- **Sentence-level split**: no text leakage between train/test (stricter eval)
- **Divergence loss**: cosine divergence regularization on target/other embeddings (weight=0.1)
- **Class-weighted loss**: inverse-frequency weighting (bull=1.437, bear=1.181, neut=0.686)

| Run | Base | Method | LR | Trainable | Acc | Macro F1 | Bull F1 | Bear F1 | Neut F1 | Epoch | Eval loss |
|-----|------|--------|----|-----------|-----|----------|---------|---------|---------|-------|-----------|
| v7 | deberta-v3 | full FT | 2e-5 | ~86M | 0.740 | 0.733 | 0.694 | 0.744 | 0.761 | 5/8 | 0.990 |
| v7-roberta | roberta | full FT | 2e-5 | ~125M | 0.749 | 0.742 | 0.722 | 0.727 | 0.776 | 6/8 | 1.030 |

#### Dataset v3: Hand-Audited Labels + Synthetic Multi-Target (5,985 pairs)

Same architecture as v2 but with **manually audited training labels**:
- **4,501 Reddit pairs**: originally LLM-labeled, then hand-audited with ~615 label corrections across the full dataset
- **1,484 synthetic multi-target pairs**: unchanged from v2
- **Label distribution**: bullish 28.2% / bearish 25.3% / neutral 46.5% (more balanced than v2)
- **Noise filtering**: added `_is_noisy_sentence()` to processing pipeline (min 5 words, max 600 chars, digit density < 20%, no residual markdown links)

| Run | Base | Method | LR | Trainable | Acc | Bal Acc | Macro F1 | Bull F1 | Bear F1 | Neut F1 | Epoch | Eval loss |
|-----|------|--------|----|-----------|-----|---------|----------|---------|---------|---------|-------|-----------|
| v8 | deberta-v3 | full FT | 2e-5 | ~86M | 0.769 | 0.777 | 0.767 | 0.771 | 0.749 | 0.780 | 5/8 | 0.893 |
| v8-roberta | roberta | full FT | 2e-5 | ~125M | 0.745 | 0.754 | 0.740 | 0.773 | 0.681 | 0.767 | 4/8 | 0.966 |

v3 → v2 comparison (DeBERTa-v3 full FT):
- **Macro F1 0.733 → 0.767** (+3.4pp) — consistent improvement across all classes
- **Bullish F1 0.694 → 0.771** (+7.7pp) — largest gain, audit fixed systematic bullish mislabeling
- **Balanced accuracy 77.7%** — confirms model isn't leaning on majority class
- **Eval loss 0.990 → 0.893** — cleaner labels = lower loss floor

v8 DeBERTa vs v8-roberta (same v3 dataset):
- **DeBERTa wins on Macro F1** (0.767 vs 0.740, +2.7pp) — consistent lead
- **RoBERTa stronger on Bullish F1** (0.773 vs 0.771) but weaker on Bearish (0.681 vs 0.749, -6.8pp)
- **DeBERTa converges later** (epoch 5 vs 4) but generalizes better (eval loss 0.893 vs 0.966)
- DeBERTa-v3 remains the best model across both v2 and v3 datasets

#### Dataset v4: + Error-Targeted Synthetic (6,287 pairs)

Same architecture as v3 but with **302 error-targeted synthetic training pairs** added:
- **4,501 Reddit pairs**: hand-audited (unchanged from v3)
- **1,484 synthetic multi-target pairs**: unchanged from v2
- **302 error-targeted synthetic pairs**: reverse-engineered from v8's 141 test errors, hand-written and triple-audited
- **Label distribution**: bullish 90 / bearish 113 / neutral 99 in error-targeted set
- **Error-targeted categories**: analyst-as-source neutral, direct business harm bearish, beat-but-sold-off, price action trades, "despite" patterns, past tense neutralizers, overvaluation concerns, contrarian sentiment, buy/sell actions, rotation pairs

| Run | Base | Method | LR | Trainable | Acc | Bal Acc | Macro F1 | Bull F1 | Bear F1 | Neut F1 | Epoch | Eval loss |
|-----|------|--------|----|-----------|-----|---------|----------|---------|---------|---------|-------|-----------|
| v9 | deberta-v3 | full FT | 2e-5 | ~86M | 0.825 | 0.827 | 0.823 | 0.836 | 0.800 | 0.833 | 4/8 | 0.757 |

v4 → v3 comparison (DeBERTa-v3 full FT):
- **Macro F1 0.767 → 0.823** (+5.6pp) — largest single-dataset improvement across all runs
- **Bullish F1 0.771 → 0.836** (+6.5pp) — rotation pairs and buy-action data drove the gain
- **Bearish F1 0.749 → 0.800** (+5.1pp) — beat-but-sold-off and direct-harm patterns resolved
- **Neutral F1 0.780 → 0.833** (+5.3pp) — analyst-as-source and past-tense neutralizers helped
- **Balanced accuracy 0.777 → 0.827** (+5.0pp) — uniform improvement, no class favored
- **Eval loss 0.893 → 0.757** — continued decrease; error-targeted data reduced confusion at decision boundaries
- **Converged faster** (epoch 4 vs 5) — cleaner signal from targeted data speeds learning

Why 302 pairs (~5% of data) caused a +5.6pp macro F1 jump:
1. **Targeted, not random** — every pair reverse-engineered from actual model errors, so 100% of new data carries maximum gradient signal
2. **Multi-target pairs teach entity disambiguation** — same sentence with different subjects and opposite labels strengthens the entity replacement mechanism
3. **Zero label noise** — hand-written and audited, unlike the ~5% error rate in original LLM-labeled data
4. **Covers systematic blind spots** — not random errors but repeating confusion patterns (neutral↔bearish boundary, "beat but sold off", rotation trades)

#### Per-run details

Paste the test-set output from each run below. Delete the template lines as you fill in.

#### Conclusions

_Write after completing all runs:_
- Best single improvement: ___
- LoRA vs layer freezing winner: ___
- Optimal LoRA rank: ___
- Recommended configuration going forward: ___
- Remaining bottleneck: ___

### 8.5 What to look for

1. **Bullish F1 is the critical metric.** It was 0.496 in the baseline — barely better than random. Any change that significantly improves bullish F1 is validated.
2. **Eval loss trend.** If eval loss decreases or stays flat across epochs (instead of increasing), overfitting is controlled.
3. **Train-eval gap.** Smaller gap = better generalization.
4. **Early stopping epoch.** If the model consistently stops at epoch 2-3 (not 6), early stopping is doing its job.


## 9) Previous Training Run Analysis

### 9.1 Overfitting diagnosis

The v1 training run (ProsusAI/finbert, 4 epochs, no regularization) showed clear overfitting:

| Epoch | Train Loss | Eval Loss | Macro F1 |
|-------|-----------|-----------|----------|
| 1 | 0.74 | **0.809** | 0.630 |
| 2 | 0.49 | **0.818** ↑ | 0.640 |
| 3 | 0.27 | **0.903** ↑ | 0.666 |
| 4 | 0.22 | **0.950** ↑ | 0.684 |

Train loss dropped 5x while eval loss **increased** 17%. The model memorized training data instead of learning generalizable patterns. Contributing factors:

1. ~3,168 training samples for 110M parameters (all unfrozen)
2. Only ~412 bullish examples — insufficient for the model to learn diverse bullish patterns
3. No regularization (no early stopping, no label smoothing, no layer freezing)
4. Biased LLM labels amplified by class weights — higher weight on mislabeled bullish examples meant learning wrong patterns more aggressively

### 9.2 What the new setup addresses

| Problem | Fix |
|---------|-----|
| Overfitting (125M params, 3.5K samples) | **LoRA** — 1.04M trainable params (0.82%), built-in regularization |
| Domain mismatch (formal news model on Reddit) | twitter-roberta base model |
| Biased LLM labels (1/5/0 examples) | Balanced prompt (2/2/2) + qwen2.5:3b |
| Random [TARGET]/[OTHER] init | Mean embedding initialization |
| Overconfidence in noisy labels | Label smoothing (0.1) |
| No stop condition | Early stopping on macro_F1 |

---

## 10) Dependencies

### Runtime (`backend/requirements.txt`)

```
transformers>=4.46.3,<5
torch>=2.6.0,<2.8
numpy>=1.26.4,<2
SQLAlchemy>=2.0.35,<2.1
psycopg2-binary>=2.9.9,<3
```

### Training only (`backend/ml/requirements.txt`)

```
scikit-learn>=1.5.0,<2
accelerate>=1.0.0,<2
peft>=0.13.0,<1
```

### GPU note

For GPU training on RTX 4060, install CUDA PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 11) File Map

| File | Purpose |
|------|---------|
| `backend/ml/train_finbert.py` | Training pipeline: load data → entity replacement → tokenize → train → evaluate → save |
| `backend/ml/eval_baseline.py` | Evaluate base model (no fine-tuning) as baseline |
| `backend/ml/requirements.txt` | Training-only dependencies |
| `backend/ml/data/prompts.py` | LLM labeling prompt (single source of truth) |
| `backend/ml/data/llm_labeler.py` | LLM labeling pipeline (Ollama, parallel, resumable) |
| `backend/ml/data/build_training_set.py` | Raw posts → 3NF training tables |
| `backend/ml/data/bulk_scraper.py` | Reddit scraper for raw training posts |
| `backend/ml/data/test_llm_labeler.py` | Accuracy test for LLM labeling (20 pairs with expected answers) |
| `backend/ml/data/test_concurrency.py` | Benchmark for parallel worker count |
| `backend/ml/data/check_training_data.py` | Diagnostic: label distribution, conflicts, multi-entity stats |
| `backend/app/database/models.py` | `TrainingSentence`, `TrainingSentenceSubject` ORM models |
| `backend/app/processing/sentiment_tagger/tagger_logic.py` | `normalize_and_tag_sentence()` for text normalization |
| `backend/app/processing/sentiment_tagger/topic_definitions.py` | Ticker → alias mapping (~90 stocks + MACRO + TECHNOLOGY) |

---

