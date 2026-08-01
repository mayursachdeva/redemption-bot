# Fibonacci-Anchored Prediction Mechanism for SL/TP and Entry

**Date:** 2026-08-01
**Status:** Approved, not yet implemented

## Problem

Stop-loss width is currently decided independently of every trend/prediction signal
the bot already computes. `compute_atr_stop_loss_pct` (execute.py:423) is pure
volatility (`ATR(14, 1h) x ATR_MULTIPLIER`, clamped to `[8%, 25%]`) — it has no
directional or structural input. Meanwhile `is_counter_trend`, `is_extended_from_vwap`,
`get_obv_divergence`, and `get_kronos_forecast` all run earlier in the same cycle to
decide *whether* to enter, but none of that informs *where* the stop or take-profit
actually sit. Take-profit is even more disconnected — a flat multiple of whatever the
stop distance happened to be (`ATR_TP_RATIO_1`/`ATR_TP_RATIO_2`, 2x/4x), with no
relationship to real price structure.

Concretely: MMT's ATR reading pushed its stop to the 25% ceiling — the clamp did its
job, but the number itself was disconnected from where the market might actually
reverse. It cost 25.41% ($764.60) on one trade purely because ATR said "volatile,"
not because any of the bot's own trend/prediction signals said "this level matters."

## Goal

Stop-loss and take-profit prices anchor to actual price structure (Fibonacci
retracement/extension levels) instead of pure percentage offsets from entry.
Fibonacci also becomes a soft input into the entry decision, alongside Kronos/LLM.

## Design

### New function: `get_fibonacci_levels` (scanner.py)

```
get_fibonacci_levels(symbol: str, is_short: bool, interval: str = "1h", period: int = 14) -> dict | None
```

Same fetch/compute split `get_atr` already uses: fetches the same 14x1h candle
window ATR already uses (no new API cost, no new timeframe concept), then hands off
to a pure function:

```
_compute_fibonacci_levels(candles: list[tuple[float, float, float]], is_short: bool) -> dict | None
```

- `swing_high` = max high over the window, `swing_low` = min low over the window.
- For a long (`is_short=False`): retracement measured **down** from `swing_high`
  toward `swing_low` (support levels below current price); extensions measured
  **beyond** `swing_high`.
- For a short (`is_short=True`): mirrored — retracement measured **up** from
  `swing_low` toward `swing_high` (resistance above current price); extensions
  **beyond** `swing_low`.
- Returns absolute prices at the standard levels:
  `retracements: {23.6, 38.2, 50, 61.8, 78.6}`, `extensions: {127.2, 161.8}`.
- Returns `None` if candles are insufficient (`< period + 1`) or the swing is
  degenerate (`swing_high <= swing_low`) — same failure contract as `get_atr`.

### New functions: SL/TP/entry (execute.py)

```
compute_fib_stop_loss_pct(symbol, price, is_short=False) -> Decimal | None
```
Distance from `price` to the 61.8% retracement level (the "golden ratio" invalidation
level), pushed a small buffer further past it so the stop doesn't sit exactly on the
level Fib traders themselves watch (same "buffer past the trigger" idea already used
in `place_oco_exit`'s `stop_price * 0.995`):

```
raw_pct = abs(price - level_61_8) / price
stop_loss_pct = raw_pct * (1 + FIB_STOP_BUFFER_PCT)  # FIB_STOP_BUFFER_PCT default 0.02 -> 2% further out
```

Result clamped to the **existing** `[ATR_MIN_STOP_LOSS_PCT, ATR_MAX_STOP_LOSS_PCT]`
bounds — no new min/max constants. Returns `None` if `get_fibonacci_levels` returns
`None`.

```
compute_fib_take_profit_pcts(symbol, price, is_short=False) -> tuple[Decimal, Decimal] | None
```
Distances from `price` to the 127.2% and 161.8% extension levels — replaces
`ATR_TP_RATIO_1`/`ATR_TP_RATIO_2` at both call sites when available. Independent of
whatever `stop_loss_pct` was chosen (unlike the current ATR-ratio approach, which
derives TP as a multiple of the stop). The existing `meets_min_reward_risk` gate
downstream still applies unchanged — if Fib's TP/SL ratio doesn't clear
`MIN_REWARD_RISK_RATIO`, the candidate is skipped exactly as today.

```
fib_entry_signal(symbol, price, is_short=False) -> Decimal
```
Small additive nudge into `compute_opportunity_score`'s new optional `fib_score`
parameter (same pattern and rough magnitude as the existing `fear_greed` nudge):
positive if price is holding above the 50%/61.8% structure (trend structurally
intact), negative if price has already broken past 61.8% (structure weakening).
Returns `Decimal("0")` (neutral, no effect) if levels are unavailable — mirrors how
`fear_greed=None` already means "no nudge" today.

### Call-site integration

`execute.py` (~line 1200) and `execute_futures.py` (~line 402), both currently:

```python
stop_loss_pct = DEFAULT_STOP_LOSS_PCT if use_fixed_stop_loss() else compute_atr_stop_loss_pct(sym, current_price)
```

become:

```python
if use_fixed_stop_loss():
    stop_loss_pct = DEFAULT_STOP_LOSS_PCT  # FUTURES_STOP_LOSS_PCT on the futures side
else:
    stop_loss_pct = compute_fib_stop_loss_pct(sym, current_price, is_short) or compute_atr_stop_loss_pct(sym, current_price)
```

and the take-profit call:

```python
tp_pcts = compute_fib_take_profit_pcts(sym, current_price, is_short) or compute_atr_take_profit_pcts(stop_loss_pct)
```

Earlier in the same per-candidate loop, alongside the existing `get_kronos_forecast`
call: `fib_score = fib_entry_signal(sym, current_price, is_short)`, passed into
`compute_opportunity_score(..., fib_score=fib_score)`.

**Fallback chain: Fib -> ATR -> flat default.** Matches the codebase's existing
graceful-degradation philosophy end to end — `compute_atr_stop_loss_pct` already
falls back to `DEFAULT_STOP_LOSS_PCT` when ATR itself is unavailable; this just adds
one more tier in front of it. Nothing about the ATR path changes; it becomes the
fallback instead of the primary.

**Fixed-SL policy interaction: unchanged precedence.** `use_fixed_stop_loss()` (the
drawdown-recovery override added 2026-07-31) still wins over both Fib and ATR while
it's active — the safety guarantee (hard-capped risk per trade during a drawdown)
stays intact regardless of what any prediction signal says. Fib only applies once
that policy self-deactivates on recovery.

**Scope:** both spot longs (`execute.py`) and futures shorts (`execute_futures.py`),
mirroring every other dual-path signal already in the codebase (VWAP deviation, OBV
divergence, counter-trend, opportunity score).

### Error handling

- `get_fibonacci_levels`: `None` on fetch failure, thin candle history, or a
  degenerate swing (zero-division guard). Same contract as `get_atr`.
- `compute_fib_stop_loss_pct` / `compute_fib_take_profit_pcts`: propagate `None`
  straight through — no exception handling needed here since `get_fibonacci_levels`
  already absorbs every failure mode internally.
- `fib_entry_signal`: never raises or returns `None` — returns `Decimal("0")`
  (neutral) on missing levels, so it always composes safely into
  `compute_opportunity_score` without special-casing at the call site.

### Testing

Same `_test_*` / `if __name__ == "__main__"` convention already used throughout
`execute.py` and `scanner.py`:

- `_test_compute_fibonacci_levels` — pure, fabricated candle tuples, no network.
  Covers: normal up-swing, normal down-swing, degenerate swing (`high == low` ->
  `None`), thin history (< period+1 candles -> `None`).
- `_test_compute_fib_stop_loss_pct` / `_test_compute_fib_take_profit_pcts` — bad
  symbol -> `get_fibonacci_levels` returns `None` -> function returns `None`, same
  minimal-mocking style as `_test_compute_atr_stop_loss_pct`.
- `_test_fib_entry_signal` — bad symbol -> neutral `Decimal("0")`.
- Extend `_test_compute_opportunity_score` to cover the new `fib_score` term
  (positive nudge raises the score, negative lowers it, zero is a no-op).

## Out of scope (explicitly deferred)

- Retiring `compute_atr_stop_loss_pct`/`compute_atr_take_profit_pcts` — they remain
  as the fallback tier, not replaced.
- Changing the fixed-SL policy's own logic (`use_fixed_stop_loss`,
  `activate_fixed_stop_loss`) — only its precedence over the new Fib path was
  confirmed (unchanged: fixed-SL still wins).
- A dedicated/longer swing-detection window separate from the ATR period — the
  same 14x1h window is reused, by design, to avoid introducing a second timeframe
  concept.
- Fib-based sizing for the laddered spot TP legs' quantity split
  (`equal_profit_fractions`) — out of scope; only the TP *percentages* change, not
  how quantity is split across legs.
