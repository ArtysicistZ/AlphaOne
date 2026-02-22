"""Quick test: does the local model distinguish "target" from "other"?

After retraining with SEntFiN-style plain-word entity replacement,
the model should give different predictions when "target" and "other"
swap positions.
"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./ml/models/deberta-v3-sentiment/best")
MODEL_PATH = parser.parse_args().model

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Verify token IDs for "target" and "other"
print(f"\nvocab_size: {len(tokenizer)}")
target_ids = tokenizer.encode("target", add_special_tokens=False)
other_ids = tokenizer.encode("other", add_special_tokens=False)
print(f"'target' -> {target_ids}")
print(f"'other'  -> {other_ids}")

# Check embedding similarity
emb = model.get_input_embeddings()
target_emb = emb.weight[target_ids[0]].detach().numpy()
other_emb = emb.weight[other_ids[0]].detach().numpy()
cosine_sim = np.dot(target_emb, other_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(other_emb))
l2_dist = np.linalg.norm(target_emb - other_emb)
print(f"\n'target' vs 'other' embedding:")
print(f"  cosine similarity: {cosine_sim:.6f}")
print(f"  L2 distance:       {l2_dist:.6f}")

# Test sentences (plain-word entity replacement)
test_cases = [
    ("target is great but other is doomed", "AAPL=target (expect bullish)"),
    ("other is great but target is doomed", "AAPL=target (expect bearish)"),
    ("target is very very good but other is very very not good", "AAPL=target (expect bullish)"),
    ("other is very very good but target is very very not good", "AAPL=target (expect bearish)"),
    ("target earnings were incredible", "single target bullish"),
    ("target is crashing hard", "single target bearish"),
]

print("\n=== Inference Results ===\n")
for text, desc in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0].detach().numpy()
    probs = np.exp(logits) / np.exp(logits).sum()
    pred_idx = int(np.argmax(probs))
    label = model.config.id2label[pred_idx]
    score = float(probs[0] - probs[1])

    print(f"  {desc}")
    print(f"    Input: {text}")
    print(f"    Label: {label}  Score: {score:+.4f}")
    print(f"    Probs: bull={probs[0]:.4f} bear={probs[1]:.4f} neut={probs[2]:.4f}")
    print()
