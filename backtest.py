"""Backtest single-TP vs laddered-TP exit strategies against real historical
daily prices. freecryptoapi's historical endpoints are Pro-only (confirmed
live: getOHLC/getHistory both reject the free-tier key) — using Binance's
own public klines instead (mainnet, no key, free).

Stop-loss/take-profit sizing mirrors execute.py's live Fib -> ATR -> flat
fallback chain (compute_fib_stop_loss_pct / compute_atr_stop_loss_pct /
DEFAULT_STOP_LOSS_PCT) instead of one hardcoded flat percentage for every
simulated entry — the two used to diverge, so this backtest was testing
numbers the live bot doesn't actually use. Computed from the pure math
functions (_compute_atr / _compute_fibonacci_levels) on a HISTORICAL window
ending at each entry, not the live get_atr()/get_fibonacci_levels() wrappers
— those only ever fetch "as of right now," which would give every simulated
day the same today's-ATR/Fib regardless of which historical day is being
tested. Constants (ATR_MULTIPLIER, the clamp bounds, FIB_STOP_BUFFER_PCT,
FIB_MAX_SCALE_FACTOR, ATR_TP_RATIO_MULTIPLIERS) are imported from execute.py
so the numbers stay in sync with live even though the control flow here is
a small, deliberate duplication of execute.py's formulas — refactoring the
live functions to accept a pre-fetched window instead of always fetching
internally was ruled out as unnecessary risk to live-path code for what
this backtest needs.
"""
from __future__ import annotations

from decimal import Decimal

import requests

from execute import (
    ATR_MAX_STOP_LOSS_PCT,
    ATR_MIN_STOP_LOSS_PCT,
    ATR_MULTIPLIER,
    ATR_TP_RATIO_MULTIPLIERS,
    DEFAULT_STOP_LOSS_PCT,
    FIB_MAX_SCALE_FACTOR,
    FIB_STOP_BUFFER_PCT,
)
from scanner import MEME_SYMBOLS, _compute_atr, _compute_fibonacci_levels, equal_profit_fractions

ATR_PERIOD = 14
FIB_PERIOD = 14
# Flat fallback tier — same role as DEFAULT_STOP_LOSS_PCT/LADDER_TP_PCTS in
# execute.py: used only when there's not enough historical window yet for
# either ATR or Fib (the first ~25 days of any candle series).
FLAT_STOP_LOSS_PCT = Decimal("0.15")
FLAT_TP_PCTS = (Decimal("0.15"), Decimal("0.30"))


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


def _window_tuples(candles: list[dict], entry_idx: int, period: int) -> list[tuple[float, float, float]] | None:
    """candles[entry_idx-period : entry_idx] as (high, low, close) float
    tuples — the shape _compute_atr/_compute_fibonacci_levels take. None if
    there's not yet a full period of history before entry_idx."""
    window = candles[max(0, entry_idx - period) : entry_idx]
    if len(window) < period:
        return None
    return [(float(c["high"]), float(c["low"]), float(c["close"])) for c in window]


def atr_stop_pct(candles: list[dict], entry_idx: int) -> Decimal | None:
    """Mirrors execute.py's compute_atr_stop_loss_pct formula exactly
    (ATR_MULTIPLIER x ATR / price, clamped to [ATR_MIN_STOP_LOSS_PCT,
    ATR_MAX_STOP_LOSS_PCT]), computed on the historical window ending at
    entry_idx. None if there's not enough history yet, or ATR itself can't
    be computed (degenerate candles) — caller falls back to Fib or flat."""
    tuples = _window_tuples(candles, entry_idx, ATR_PERIOD)
    if tuples is None:
        return None
    atr = _compute_atr(tuples, ATR_PERIOD)
    if atr is None:
        return None
    price = float(candles[entry_idx]["close"])
    if price <= 0:
        return None
    raw_pct = Decimal(str(atr / price)) * ATR_MULTIPLIER
    return max(ATR_MIN_STOP_LOSS_PCT, min(ATR_MAX_STOP_LOSS_PCT, raw_pct))


def fib_stop_and_tp_pcts(candles: list[dict], entry_idx: int) -> tuple[Decimal, tuple[Decimal, Decimal]] | None:
    """Mirrors execute.py's compute_fib_stop_loss_pct/compute_fib_take_profit_pcts:
    stop at the 23.6% retracement (+ FIB_STOP_BUFFER_PCT, clamped to the ATR
    bounds), take-profit legs at the 161.8%/261.8% extensions scaled by
    however much the stop's clamp stretched it, capped at FIB_MAX_SCALE_FACTOR.
    None (no Fib available, or the scale would exceed the cap) — caller
    falls back to ATR."""
    tuples = _window_tuples(candles, entry_idx, FIB_PERIOD)
    if tuples is None:
        return None
    levels = _compute_fibonacci_levels(tuples, is_short=False, period=FIB_PERIOD)
    if levels is None:
        return None
    price = Decimal(str(candles[entry_idx]["close"]))
    if price <= 0:
        return None
    level_23_6 = Decimal(str(levels["retracements"][23.6]))
    raw_pct = abs(price - level_23_6) / price
    buffered_pct = raw_pct * (1 + FIB_STOP_BUFFER_PCT)
    clamped_pct = max(ATR_MIN_STOP_LOSS_PCT, min(ATR_MAX_STOP_LOSS_PCT, buffered_pct))
    scale = clamped_pct / buffered_pct if buffered_pct > 0 else Decimal("1")
    if scale > FIB_MAX_SCALE_FACTOR:
        return None
    tp1 = Decimal(str(levels["extensions"][161.8]))
    tp2 = Decimal(str(levels["extensions"][261.8]))
    return (clamped_pct, (abs(tp1 - price) / price * scale, abs(tp2 - price) / price * scale))


def sourced_stop_and_tp(candles: list[dict], entry_idx: int) -> tuple[Decimal, tuple[Decimal, Decimal]]:
    """Fib -> ATR -> flat fallback chain — same precedence execute.py's live
    call sites use. This is the actual parity fix: every simulated entry now
    gets stop/TP sized the way the live bot would size it that day, instead
    of one flat percentage for the whole backtest."""
    fib = fib_stop_and_tp_pcts(candles, entry_idx)
    if fib is not None:
        return fib
    atr_stop = atr_stop_pct(candles, entry_idx)
    if atr_stop is not None:
        tp1 = atr_stop * ATR_TP_RATIO_MULTIPLIERS[0]
        tp2 = atr_stop * ATR_TP_RATIO_MULTIPLIERS[1]
        return (atr_stop, (tp1, tp2))
    return (FLAT_STOP_LOSS_PCT, FLAT_TP_PCTS)


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
    days (they need room to resolve against future candles). Stop/TP are
    sourced per-entry via sourced_stop_and_tp — real historical Fib/ATR
    sizing, not one flat percentage for the whole run."""
    single_returns, ladder_returns = [], []
    usable = max(len(candles) - lookahead_buffer, 0)
    for i in range(usable):
        stop_pct, tp_pcts = sourced_stop_and_tp(candles, i)
        single_returns.append(simulate_exit(candles, i, tp_pcts[0], stop_pct))
        legs = tuple(zip(equal_profit_fractions(tp_pcts), tp_pcts))
        leg_returns = [simulate_exit(candles, i, tp, stop_pct) for _, tp in legs]
        ladder_returns.append(sum(f * r for (f, _), r in zip(legs, leg_returns)))
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


def _test_sourced_stop_and_tp() -> None:
    # thin history (< max(ATR_PERIOD, FIB_PERIOD) candles before entry_idx) -> flat fallback
    thin = [_candle(100, 100, 100) for _ in range(5)]
    stop_pct, tp_pcts = sourced_stop_and_tp(thin, 3)
    assert (stop_pct, tp_pcts) == (FLAT_STOP_LOSS_PCT, FLAT_TP_PCTS), (stop_pct, tp_pcts)

    # enough history, flat price action (zero true range) -> ATR is 0, Fib
    # swing is degenerate (high==low every candle) -> both None -> flat fallback
    flat_series = [_candle(100, 100, 100) for _ in range(30)]
    stop_pct, tp_pcts = sourced_stop_and_tp(flat_series, 25)
    assert (stop_pct, tp_pcts) == (FLAT_STOP_LOSS_PCT, FLAT_TP_PCTS), (stop_pct, tp_pcts)

    # a real swing in the 14-candle window immediately before entry_idx,
    # with the entry candle itself sitting well clear of the 23.6%
    # retracement level (so the stop's clamp-to-floor scale factor stays
    # under FIB_MAX_SCALE_FACTOR instead of triggering the cap) -> Fib
    # should produce a usable (stop, tp_pcts) pair, preferred over ATR
    window = [_candle(100, 90, 95) for _ in range(5)] + [_candle(110, 100, 105) for _ in range(9)]
    swing = window + [_candle(99, 97, 98)]  # entry candle (index 14), not part of the lookback window
    stop_pct, tp_pcts = sourced_stop_and_tp(swing, 14)
    assert stop_pct > 0
    assert tp_pcts[0] > 0 and tp_pcts[1] > tp_pcts[0]
    direct = fib_stop_and_tp_pcts(swing, 14)
    assert direct is not None and (stop_pct, tp_pcts) == direct, "Fib should win over ATR when both are available"


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


# (fetch_days, lookahead_buffer_candles) per timeframe — buffer is ~24h of
# room for TP/SL to resolve, converted to candle count for that granularity
_TIMEFRAME_CONFIG = {
    "1d": {"days": 400, "buffer": 60},       # 60 days room (matches original backtest)
    "1m": {"days": 5, "buffer": 1440},       # 5 days of 1m bars, 24h buffer
    "5m": {"days": 20, "buffer": 288},       # 20 days of 5m bars, 24h buffer
}


if __name__ == "__main__":
    import sys

    _test_simulate_exit()  # fails loudly if the walk-forward logic breaks
    _test_sourced_stop_and_tp()
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1d"
    if timeframe not in _TIMEFRAME_CONFIG:
        raise SystemExit(f"Unknown timeframe {timeframe!r} — use one of {list(_TIMEFRAME_CONFIG)}")
    run_backtest(timeframe)
