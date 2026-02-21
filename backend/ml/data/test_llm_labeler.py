"""
Quick test: send 10 sample sentence-subject pairs to the local Ollama model
(one pair per call) and print the labeling results.

Usage:
    cd backend
    python -m ml.data.test_llm_labeler
    python -m ml.data.test_llm_labeler --model qwen2.5:1.5b
"""

import json
import re
import argparse
import time

import requests

from ml.data.prompts import SENTIMENT_SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:1.5b"

# Test pairs covering original cases + known failure modes
TEST_PAIRS = [
    # ── Original cases ─────────────────────────────────────────
    (1,  "NVDA", "It will pass NVDA this year and it's over 4T already"),
    (2,  "COIN", "It's not a hedge of inflation, it's not better than fartcoin or retard COIN All"),
    (3,  "AAPL", "AAPL is nowhere close to the best stock"),
    (4,  "MSFT", "Plan is to sell for my profit and roll it into an MSFT leap and more GLD."),
    (5,  "MSFT", "MSFT cant even get representation in a bubble burst meme."),
    (6,  "AMZN", "I trade for fun but if I really wanted results I'd probably just accumulate AMZN and hold for life"),
    (7,  "BLK",  "Oh em geee 4 yr cycle, blah blah Naw retards, BLK got that shit pinned Dumb it to zero finally so it can stop dicking my other stocks"),
    (8,  "SCHW", "SCHW only shows up until the close, after hours I lost another few hundred grand and am sitting at 1.4M now with more option losses hitting when market opens."),
    (9,  "AMZN", "AMZN MSFT GOOG NVDA."),
    (10, "GOOG", "AMZN MSFT GOOG NVDA."),
    # ── Options trades (puts = bearish, calls = bullish) ───────
    (11, "NVDA", "bought a ton of weekly far OTM puts on NVDA and woke up today with huge profits!"),
    (12, "GOOG", "GOOG Puts $30000 Gains Up 150%"),
    (13, "AAPL", "Hit +100% on AAPL puts, rolled that into TD puts for another +100%."),
    (14, "TSLA", "At 10:47am I bought 32 contracts of TSLA $235.00 Put @ $279.00 per contract."),
    (15, "MSFT", "100k MSFT Calls. Way oversold, probably see 430 before my expiry."),
    # ── Sarcasm / irony ────────────────────────────────────────
    (16, "GS",   "Well if GS says that's what's gonna happen, that must be what's gonna happen."),
    # ── Flat / going nowhere = NOT bullish ─────────────────────
    (17, "INTC", "$10,000 of INTC bought 25 years ago is worth $10,000 today."),
    # ── Clearly bearish mislabeled as bullish ──────────────────
    (18, "TSLA", "TSLA will go to 80...all big tech obliterated."),
    (19, "TSLA", "I think we could actually see sub $100 pps for TSLA in the near future."),
    (20, "PLTR", "PLTR will see 95-105 this year before it bounces back"),
]

# What a human would label. Tuples = multiple acceptable answers (first is preferred).
EXPECTED = {
    1:  ("bearish",),           # "pass NVDA" = surpass/overtake NVDA → bearish on NVDA
    2:  ("bearish",),           # "not better than fartcoin" → bearish on COIN
    3:  ("neutral", "bearish"), # "nowhere close to the best" — mild opinion, borderline
    4:  ("bullish",),           # buying MSFT leaps → bullish on MSFT
    5:  ("bearish",),           # "can't even get representation" → bearish on MSFT
    6:  ("bullish",),           # "accumulate and hold for life" → bullish on AMZN
    7:  ("bearish",),           # "dumb it to zero" → bearish on BLK
    8:  ("bearish",),           # "lost another few hundred grand" → bearish on SCHW
    9:  ("neutral",),           # just listing tickers → neutral
    10: ("neutral",),           # just listing tickers → neutral
    11: ("bearish",),           # buying PUTS on NVDA = bearish on NVDA
    12: ("bearish",),           # GOOG puts profit = bearish on GOOG
    13: ("bearish",),           # AAPL puts profit = bearish on AAPL
    14: ("bearish",),           # buying TSLA puts = bearish on TSLA
    15: ("bullish",),           # buying MSFT calls = bullish on MSFT
    16: ("neutral", "bearish"), # sarcasm about GS predictions — borderline
    17: ("bearish", "neutral"), # flat for 25 years — negative but arguably factual
    18: ("bearish",),           # "will go to 80" = bearish on TSLA
    19: ("bearish",),           # "sub $100" = bearish on TSLA
    20: ("bearish",),           # "see 95-105 before it bounces back" = bearish near-term
}


def call_single(model: str, subject: str, sentence: str) -> dict | None:
    """Send one pair to Ollama, return parsed result or None."""
    user_msg = f'subject="{subject}", sentence="{sentence}"\n/no_think'
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256},
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"].strip()

    # Strip Qwen3 <think>...</think> tags and markdown fences
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"raw": raw, "label": None, "confidence": 0}

    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"raw": raw, "label": None, "confidence": 0}

    label = obj.get("label", "").lower().strip()
    if label not in ("bullish", "bearish", "neutral"):
        return {"raw": raw, "label": None, "confidence": 0}

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {"raw": raw, "label": label, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Test LLM labeler with sample pairs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Testing {len(TEST_PAIRS)} pairs (one call each)...")
    print()

    try:
        # Warm up: check connection
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama. Is it running?")
        return

    results = {}
    total_time = 0
    for pid, subj, sent in TEST_PAIRS:
        t0 = time.time()
        try:
            result = call_single(args.model, subj, sent)
            elapsed = time.time() - t0
            total_time += elapsed
            results[pid] = result
            status = result["label"] or "FAIL"
            print(f"  [{pid:>2}] {status:<8} ({elapsed:.1f}s) {subj}: {sent[:50]}...")
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            results[pid] = {"label": None, "confidence": 0, "raw": str(e)}
            print(f"  [{pid:>2}] ERROR   ({elapsed:.1f}s) {subj}: {e}")

    print()
    print("=== RESULTS ===")
    print(f"{'ID':<4} {'Subject':<8} {'LLM':<10} {'Expected':<18} {'Match':<6} {'Conf':<6} Sentence")
    print("-" * 120)

    correct = 0
    total = 0
    for pid, subj, sent in TEST_PAIRS:
        r = results.get(pid, {})
        label = r.get("label") or "MISS"
        conf = r.get("confidence", 0)
        acceptable = EXPECTED.get(pid, ("?",))
        matched = label in acceptable
        if matched:
            correct += 1
        total += 1
        expected_str = acceptable[0] if len(acceptable) == 1 else "/".join(acceptable)
        mark = "Y" if matched else "N"
        print(f"{pid:<4} {subj:<8} {label:<10} {expected_str:<18} {mark:<6} {conf:<6.2f} {sent[:55]}")

    print("-" * 120)
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time / len(TEST_PAIRS):.1f}s per pair)")
    print()

    if correct / total >= 0.8:
        print("PASS - Model is suitable for labeling.")
    elif correct / total >= 0.6:
        print("MARGINAL - Consider using a larger model.")
    else:
        print("FAIL - Model is not reliable enough. Try a larger model.")


if __name__ == "__main__":
    main()
