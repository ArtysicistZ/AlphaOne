"""
LLM-based sentiment labeler using a local Ollama model.

Sends ONE (sentence, subject) pair per LLM call for reliability,
but runs multiple calls in PARALLEL via ThreadPoolExecutor for speed.

Resumable: only processes rows where sentiment_label IS NULL.
Commits every N labels for checkpoint safety.

Prerequisites:
    1. Install Ollama:  https://ollama.com/download
    2. Pull model:      ollama pull qwen2.5:1.5b
    3. Set parallel:    set OLLAMA_NUM_PARALLEL=6  (then restart Ollama)

Usage:
    cd backend
    python -m ml.data.llm_labeler
    python -m ml.data.llm_labeler --workers 6 --model qwen2.5:1.5b
"""

import json
import logging
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sqlalchemy import func as sql_func

from ml.data.prompts import SENTIMENT_SYSTEM_PROMPT
from app.database.session import init_db, SessionLocal
from app.database.models import TrainingSentence, TrainingSentenceSubject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_WORKERS = 4
COMMIT_EVERY = 20


def call_ollama(model: str, subject: str, sentence: str, timeout: int = 60) -> str:
    """Send a single (subject, sentence) pair to Ollama, return raw response text."""
    user_msg = f'subject="{subject}", sentence="{sentence}"'
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 64,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_single_response(raw_text: str) -> dict | None:
    """Parse a single {"label": ..., "confidence": ...} from LLM output."""
    text = raw_text.strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    label = obj.get("label", "").lower().strip()
    if label not in ("bullish", "bearish", "neutral"):
        return None

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {"label": label, "confidence": confidence}


def label_one(model: str, pair: dict) -> dict:
    """Label a single pair (thread-safe, no DB access). Returns result dict."""
    try:
        raw = call_ollama(model, pair["subject"], pair["text"])
        result = parse_single_response(raw)
        if result:
            return {"id": pair["id"], "ok": True, **result}
        else:
            return {"id": pair["id"], "ok": False, "raw": raw[:100]}
    except Exception as e:
        return {"id": pair["id"], "ok": False, "raw": str(e)[:100]}


def fetch_unlabeled(db, limit: int) -> list[dict]:
    """Fetch unlabeled (sentence, subject) pairs."""
    rows = (
        db.query(
            TrainingSentenceSubject.id,
            TrainingSentenceSubject.subject,
            TrainingSentence.normalized_text,
        )
        .join(TrainingSentence, TrainingSentenceSubject.sentence_id == TrainingSentence.id)
        .filter(TrainingSentenceSubject.sentiment_label.is_(None))
        .order_by(TrainingSentenceSubject.id.asc())
        .limit(limit)
        .all()
    )
    return [{"id": r[0], "subject": r[1], "text": r[2]} for r in rows]


def run_labeler(
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    commit_every: int = COMMIT_EVERY,
    max_labels: int = 0,
) -> dict:
    """Main labeling loop. Parallel LLM calls, sequential DB writes."""
    init_db()
    db = SessionLocal()

    total_remaining = (
        db.query(sql_func.count(TrainingSentenceSubject.id))
        .filter(TrainingSentenceSubject.sentiment_label.is_(None))
        .scalar()
    )
    logger.info(
        "llm_labeler_started model=%s workers=%d unlabeled=%d",
        model, workers, total_remaining,
    )

    total_labeled = 0
    total_failed = 0
    uncommitted = 0
    t_start = time.time()

    try:
        while True:
            if max_labels > 0 and total_labeled >= max_labels:
                break

            pairs = fetch_unlabeled(db, limit=50)
            if not pairs:
                break

            # Fire off parallel LLM calls
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(label_one, model, pair): pair
                    for pair in pairs
                }

                for future in as_completed(futures):
                    result = future.result()

                    if result["ok"]:
                        db.query(TrainingSentenceSubject).filter(
                            TrainingSentenceSubject.id == result["id"]
                        ).update({
                            "sentiment_label": result["label"],
                            "sentiment_confidence": result["confidence"],
                        })
                        total_labeled += 1
                        uncommitted += 1

                        if uncommitted >= commit_every:
                            db.commit()
                            elapsed = time.time() - t_start
                            rate = total_labeled / elapsed
                            remaining = total_remaining - total_labeled - total_failed
                            eta_min = (remaining / rate / 60) if rate > 0 else 0
                            logger.info(
                                "progress labeled=%d failed=%d remaining~=%d rate=%.1f/s eta=%.0fmin",
                                total_labeled, total_failed, max(0, remaining), rate, eta_min,
                            )
                            uncommitted = 0
                    else:
                        total_failed += 1
                        logger.warning("failed id=%d raw=%s", result["id"], result.get("raw", ""))

        if uncommitted > 0:
            db.commit()

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving progress...")
        if uncommitted > 0:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    elapsed = time.time() - t_start
    result = {
        "total_labeled": total_labeled,
        "total_failed": total_failed,
        "elapsed_min": round(elapsed / 60, 1),
    }
    logger.info("llm_labeler_completed %s", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Label training pairs with local LLM via Ollama")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel Ollama calls")
    parser.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to label (0=all)")
    args = parser.parse_args()

    print(f"=== LLM Labeler (Ollama, {args.workers} workers) ===")
    print(f"Model: {args.model}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print()

    result = run_labeler(model=args.model, workers=args.workers, commit_every=args.commit_every, max_labels=args.limit)

    print()
    print("=== Results ===")
    print(f"Total labeled:  {result['total_labeled']}")
    print(f"Total failed:   {result['total_failed']}")
    print(f"Elapsed:        {result['elapsed_min']} min")


if __name__ == "__main__":
    main()
