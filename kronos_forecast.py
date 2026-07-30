"""Kronos foundation-model price forecast for one symbol. Runs in its own
venv (kronos_venv, Python 3.11+, needs torch) — separate from the main
scanner venv (Python 3.9) since Kronos requires Python 3.10+. Invoked as a
subprocess from scanner.py (see get_kronos_forecast), not imported directly.

Usage: kronos_venv/bin/python kronos_forecast.py SYMBOL [interval] [pred_len]
Prints one JSON line to stdout.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kronos_repo"))

import pandas as pd
import requests
from model import Kronos, KronosPredictor, KronosTokenizer

_predictor = None


def _get_predictor() -> KronosPredictor:
    global _predictor
    if _predictor is None:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        # ponytail: MPS (Apple GPU) backend hits a PyTorch limitation here
        # (scaled_dot_product_attention + dropout unsupported on MPS) — force
        # CPU. Model is only 24.7M params, CPU inference is fine for one
        # symbol per bot cycle.
        _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    return _predictor


def get_klines_df(symbol: str, interval: str = "5m", lookback: int = 400) -> pd.DataFrame:
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": f"{symbol}USDT", "interval": interval, "limit": lookback},
        timeout=10,
    )
    resp.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(resp.json(), columns=cols)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def forecast_symbol(symbol: str, interval: str = "5m", lookback: int = 400, pred_len: int = 12) -> dict:
    """Kronos forecast: predicted % change of close price `pred_len`
    candles ahead vs the last real close."""
    predictor = _get_predictor()
    df = get_klines_df(symbol, interval, lookback)
    step = df["timestamps"].iloc[-1] - df["timestamps"].iloc[-2]
    y_timestamp = pd.Series([df["timestamps"].iloc[-1] + step * (i + 1) for i in range(pred_len)])

    pred_df = predictor.predict(
        df=df[["open", "high", "low", "close", "volume"]],
        x_timestamp=df["timestamps"],
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    )
    current_close = float(df["close"].iloc[-1])
    predicted_close = float(pred_df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "interval": interval,
        "pred_len": pred_len,
        "current_close": current_close,
        "predicted_close": predicted_close,
        "predicted_pct_change": (predicted_close / current_close - 1) * 100,
    }


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "DOGE"
    interval = sys.argv[2] if len(sys.argv) > 2 else "5m"
    pred_len = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print(json.dumps(forecast_symbol(symbol, interval, pred_len=pred_len)))
