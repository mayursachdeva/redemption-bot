"""FinBERT (ProsusAI/finbert) text sentiment for one string. Runs in its own
venv (kronos_venv, Python 3.11+, needs torch + transformers) — same venv
Kronos uses, since Kronos already requires Python 3.10+ and torch there.
Invoked as a subprocess from scanner.py (see get_text_sentiment), not
imported directly.

Usage: kronos_venv/bin/python finbert_sentiment.py "some text"
Prints one JSON line to stdout: {"positive": p, "negative": n, "neutral": u,
"compound": p - n}. compound is -1 (bearish) to +1 (bullish), same scale
convention as VADER's compound score so callers don't need to change.
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
    return _tokenizer, _model


def score_text(text: str) -> dict:
    tokenizer, model = _get_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
    positive, negative, neutral = (float(p) for p in probs)
    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "compound": positive - negative,
    }


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(score_text(text)))
