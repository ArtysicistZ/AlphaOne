"""Quick test: does the local model distinguish [TARGET] from [OTHER]?"""

import torch
import numpy as np
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./ml/models/deberta-v3-lr1e5/best"

print("Loading model from local path...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Verify special tokens
print(f"\nvocab_size: {len(tokenizer)}")
print(f"[TARGET] -> {tokenizer.encode('[TARGET]', add_special_tokens=False)}")
print(f"[OTHER]  -> {tokenizer.encode('[OTHER]', add_special_tokens=False)}")

# Check embedding similarity
emb = model.get_input_embeddings()
target_emb = emb.weight[128001].detach().numpy()
other_emb = emb.weight[128002].detach().numpy()
cosine_sim = np.dot(target_emb, other_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(other_emb))
l2_dist = np.linalg.norm(target_emb - other_emb)
print(f"\n[TARGET] vs [OTHER] embedding:")
print(f"  cosine similarity: {cosine_sim:.6f}")
print(f"  L2 distance:       {l2_dist:.6f}")

# Test sentences
test_cases = [
    ("[TARGET] is great but [OTHER] is doomed", "AAPL=TARGET"),
    ("[OTHER] is great but [TARGET] is doomed", "TSLA=TARGET"),
    ("[TARGET] is very very good but [OTHER] is very very not good", "AAPL=TARGET"),
    ("[OTHER] is very very good but [TARGET] is very very not good", "TSLA=TARGET"),
    ("[TARGET] earnings were incredible", "single target bullish"),
    ("[TARGET] is crashing hard", "single target bearish"),
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
