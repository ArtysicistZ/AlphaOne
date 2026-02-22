import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from app.processing.nlp_processor import get_sentiment_batch, _get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AlphaOne Inference API", version="1.0.0")


@app.on_event("startup")
def _debug_tokenizer():
    """Log tokenizer handling of special tokens on startup."""
    tokenizer, _ = _get_model()
    for token in ["[TARGET]", "[OTHER]"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        pieces = tokenizer.tokenize(token)
        logger.info("TOKENIZER DEBUG: %s -> ids=%s, pieces=%s", token, ids, pieces)
    logger.info("TOKENIZER DEBUG: vocab_size=%d", len(tokenizer))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

TARGET_TOKEN = "[TARGET]"
OTHER_TOKEN = "[OTHER]"


def _replace_targets(text: str, target: str, all_targets: list[str]) -> str:
    """Replace the target word with [TARGET] and all other targets with [OTHER]."""
    result = text
    # Replace longer targets first to avoid partial matches
    for t in sorted(all_targets, key=len, reverse=True):
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        if t.lower() == target.lower():
            result = pattern.sub(TARGET_TOKEN, result)
        else:
            result = pattern.sub(OTHER_TOKEN, result)
    return result


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    targets: list[str] = Field(..., min_length=1)


class SentimentResult(BaseModel):
    sentence: str
    normalizedInput: str
    target: str
    sentiment: str
    score: float


class InferenceResponse(BaseModel):
    results: list[SentimentResult]


@app.post("/api/v1/inference", response_model=InferenceResponse)
def run_inference(req: InferenceRequest):
    sentence = req.text.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence cannot be empty.")

    targets = [t.strip() for t in req.targets if t.strip()]
    if not targets:
        raise HTTPException(status_code=400, detail="At least one target is required.")

    # For each target, build the replaced text
    replaced_texts = [_replace_targets(sentence, t, targets) for t in targets]
    predictions = get_sentiment_batch(replaced_texts)

    results = []
    for target, replaced, (score, label) in zip(targets, replaced_texts, predictions):
        results.append(SentimentResult(
            sentence=sentence,
            normalizedInput=replaced,
            target=target,
            sentiment=label,
            score=round(score, 4),
        ))

    return InferenceResponse(results=results)
