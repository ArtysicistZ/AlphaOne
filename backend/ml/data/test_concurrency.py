"""
Benchmark: sequential vs parallel Ollama calls.
Tests 1, 2, and 4 workers on the same 10 pairs.

Usage:
    cd backend
    python -m ml.data.test_concurrency
    python -m ml.data.test_concurrency --model qwen2.5:1.5b
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ml.data.prompts import SENTIMENT_SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"

TEST_PAIRS = [
    ("NVDA", "It will pass NVDA this year and it's over 4T already"),
    ("COIN", "It's not a hedge of inflation, it's not better than fartcoin or retard COIN All"),
    ("AAPL", "AAPL is nowhere close to the best stock"),
    ("MSFT", "Plan is to sell for my profit and roll it into an MSFT leap and more GLD."),
    ("MSFT", "MSFT cant even get representation in a bubble burst meme."),
    ("AMZN", "I trade for fun but if I really wanted results I'd probably just accumulate AMZN and hold for life"),
    ("BLK",  "Oh em geee 4 yr cycle, blah blah Naw retards, BLK got that shit pinned"),
    ("SCHW", "SCHW only shows up until the close, after hours I lost another few hundred grand"),
    ("AMZN", "AMZN MSFT GOOG NVDA."),
    ("GOOG", "AMZN MSFT GOOG NVDA."),
]


def call_once(model: str, subject: str, sentence: str) -> float:
    """Make one Ollama call, return elapsed seconds."""
    user_msg = f'subject="{subject}", sentence="{sentence}"'
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 64},
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return time.time() - t0


def bench(model: str, workers: int) -> float:
    """Run all 10 pairs with N workers, return total wall time."""
    t0 = time.time()

    if workers == 1:
        for subj, sent in TEST_PAIRS:
            call_once(model, subj, sent)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(call_once, model, subj, sent) for subj, sent in TEST_PAIRS]
            for f in as_completed(futures):
                f.result()

    return time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Pairs: {len(TEST_PAIRS)}")
    print()

    # Warm up (first call loads model into GPU)
    print("Warming up...")
    call_once(args.model, "AAPL", "test")
    print()

    for w in [2, 4, 6, 8]:
        elapsed = bench(args.model, w)
        per_pair = elapsed / len(TEST_PAIRS)
        speedup = "" if w == 1 else f" (projected: {3500 * per_pair / 60:.0f} min for 3.5k pairs)"
        print(f"Workers={w}: {elapsed:.1f}s total, {per_pair:.2f}s/pair{speedup}")

    # Also show projected times for 3.5k pairs
    print()
    '''
    seq_time = bench(args.model, 1)
    seq_per = seq_time / len(TEST_PAIRS)
    print(f"--- Projected for 3,500 pairs ---")
    for w in [1, 2, 4]:
        t = bench(args.model, w)
        per = t / len(TEST_PAIRS)
        mins = 3500 * per / 60
        print(f"Workers={w}: ~{mins:.0f} min")
    '''


if __name__ == "__main__":
    main()
