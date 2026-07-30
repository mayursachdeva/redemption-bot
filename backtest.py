"""Backtest single-TP vs laddered-TP exit strategies against real historical
daily prices. freecryptoapi's historical endpoints are Pro-only (confirmed
live: getOHLC/getHistory both reject the free-tier key) — using Binance's
own public klines instead (mainnet, no key, free).
"""
from __future__ import annotations

from decimal import Decimal

import requests

from scanner import MEME_SYMBOLS, equal_profit_fractions

STOP_LOSS_PCT = Decimal("0.15")
SINGLE_TP_PCT = Decimal("0.30")
LADDER_TP_PCTS = (Decimal("0.15"), Decimal("0.30"))
# equal-profit weighted, not equal-quantity — matches execute.py's
# place_gamble_trade_laddered (a farther target gets a smaller slice)
LADDER_LEGS = tuple(zip(equal_profit_fractions(LADDER_TP_PCTS), LADDER_TP_PCTS))


def get_intraday_klines(symbol: str, interval: str, days: float) -> list[dict]:
    """Paginated fetch — Binance caps klines at 1000 candles/request, so
    intraday windows need multiple calls chained backwards via endTime."""
    interval_minutes = {"1m": 1, "5m": 5}[interval]
    target = int(days * 24 * 60 / interval_minutes)
    candles: list[dict] = []
    end_time = None
    while len(candles) < target:
        params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        candles = [
            {"high": Decimal(k[2]), "low": Decimal(k[3]), "close": Decimal(k[4])} for k in batch
        ] + candles
        end_time = batch[0][0] - 1
        if len(batch) < 1000:
            break  # hit the start of available history
    return candles[-target:] if len(candles) > target else candles


def get_daily_klines(symbol: str, days: int = 400) -> list[dict]:
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": f"{symbol}USDT", "interval": "1d", "limit": days},
        timeout=10,
    )
    resp.raise_for_status()
    return [
        {"high": Decimal(k[2]), "low": Decimal(k[3]), "close": Decimal(k[4])}
        for k in resp.json()
    ]


def simulate_exit(candles: list[dict], entry_idx: int, tp_pct: Decimal, sl_pct: Decimal) -> Decimal:
    """Walk forward from entry_idx+1 until TP or SL is touched. If both are
    touched on the same day, SL wins — daily bars can't tell intraday order,
    so assume the pessimistic case. If neither is hit before data runs out,
    mark-to-market at the last available close."""
    entry_price = candles[entry_idx]["close"]
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    for day in candles[entry_idx + 1 :]:
        if day["low"] <= sl_price:
            return -sl_pct
        if day["high"] >= tp_price:
            return tp_pct
    return candles[-1]["close"] / entry_price - 1


def backtest_symbol(candles: list[dict], lookahead_buffer: int = 60) -> dict:
    """One simulated entry per day, excluding the last `lookahead_buffer`
    days (they need room to resolve against future candles)."""
    single_returns, ladder_returns = [], []
    usable = max(len(candles) - lookahead_buffer, 0)
    for i in range(usable):
        single_returns.append(simulate_exit(candles, i, SINGLE_TP_PCT, STOP_LOSS_PCT))
        leg_returns = [simulate_exit(candles, i, tp, STOP_LOSS_PCT) for _, tp in LADDER_LEGS]
        ladder_returns.append(sum(f * r for (f, _), r in zip(LADDER_LEGS, leg_returns)))
    return {"single": single_returns, "ladder": ladder_returns}


def summarize(returns: list[Decimal]) -> dict:
    n = len(returns)
    wins = sum(1 for r in returns if r > 0)
    return {
        "n": n,
        "win_rate_pct": float(wins / n * 100) if n else 0.0,
        "avg_return_pct": float(sum(returns) / n * 100) if n else 0.0,
    }


def _candle(high, low, close) -> dict:
    return {"high": Decimal(high), "low": Decimal(low), "close": Decimal(close)}


def _test_simulate_exit() -> None:
    # entry at close=100: TP hit day 2 (high=131), SL not touched first
    up = [_candle(100, 100, 100), _candle(110, 95, 105), _candle(131, 120, 130)]
    assert simulate_exit(up, 0, Decimal("0.3"), Decimal("0.15")) == Decimal("0.3")

    # entry at close=100: SL hit day 1 (low=84)
    down = [_candle(100, 100, 100), _candle(105, 84, 90)]
    assert simulate_exit(down, 0, Decimal("0.3"), Decimal("0.15")) == Decimal("-0.15")

    # both thresholds crossed same day — SL wins (conservative)
    both = [_candle(100, 100, 100), _candle(140, 80, 100)]
    assert simulate_exit(both, 0, Decimal("0.3"), Decimal("0.15")) == Decimal("-0.15")

    # neither hit, data runs out — mark to market
    flat = [_candle(100, 100, 100), _candle(105, 98, 102)]
    assert simulate_exit(flat, 0, Decimal("0.3"), Decimal("0.15")) == Decimal("0.02")


# (fetch_days, lookahead_buffer_candles) per timeframe — buffer is ~24h of
# room for TP/SL to resolve, converted to candle count for that granularity
_TIMEFRAME_CONFIG = {
    "1d": {"days": 400, "buffer": 60},       # 60 days room (matches original backtest)
    "1m": {"days": 5, "buffer": 1440},       # 5 days of 1m bars, 24h buffer
    "5m": {"days": 20, "buffer": 288},       # 20 days of 5m bars, 24h buffer
}


def run_backtest(timeframe: str) -> None:
    cfg = _TIMEFRAME_CONFIG[timeframe]
    print(f"=== Timeframe: {timeframe} ({cfg['days']} days of history) ===")
    print(f"{'SYMBOL':6s} {'N':>5s} {'Single WinRate':>15s} {'Single Avg%':>12s} {'Ladder WinRate':>15s} {'Ladder Avg%':>12s}")
    all_single: list[Decimal] = []
    all_ladder: list[Decimal] = []
    for sym in MEME_SYMBOLS:
        candles = (
            get_daily_klines(sym, days=cfg["days"])
            if timeframe == "1d"
            else get_intraday_klines(sym, timeframe, days=cfg["days"])
        )
        result = backtest_symbol(candles, lookahead_buffer=cfg["buffer"])
        s = summarize(result["single"])
        l = summarize(result["ladder"])
        all_single += result["single"]
        all_ladder += result["ladder"]
        print(f"{sym:6s} {s['n']:5d} {s['win_rate_pct']:14.1f}% {s['avg_return_pct']:11.2f}% "
              f"{l['win_rate_pct']:14.1f}% {l['avg_return_pct']:11.2f}%")

    print(f"\n--- OVERALL ({timeframe}, equal-weight, no compounding) ---")
    s, l = summarize(all_single), summarize(all_ladder)
    print(f"Single-TP:  n={s['n']:4d}  win_rate={s['win_rate_pct']:.1f}%  avg_return/trade={s['avg_return_pct']:.2f}%")
    print(f"Laddered:   n={l['n']:4d}  win_rate={l['win_rate_pct']:.1f}%  avg_return/trade={l['avg_return_pct']:.2f}%")


if __name__ == "__main__":
    import sys

    _test_simulate_exit()  # fails loudly if the walk-forward logic breaks
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1d"
    if timeframe not in _TIMEFRAME_CONFIG:
        raise SystemExit(f"Unknown timeframe {timeframe!r} — use one of {list(_TIMEFRAME_CONFIG)}")
    run_backtest(timeframe)
