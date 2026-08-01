# Fibonacci-Anchored Prediction Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anchor stop-loss and take-profit to real price structure (Fibonacci retracement/extension levels) instead of pure ATR volatility, and add Fibonacci as a soft entry signal — fixing the disconnect where stop width had no relationship to any trend/prediction signal the bot already computes.

**Architecture:** A new `get_fibonacci_levels` in `scanner.py` (same fetch/pure-compute split as `get_atr`) computes retracement/extension prices from the same 14x1h candle window ATR already uses. Three new consumer functions in `execute.py` read from it: `compute_fib_stop_loss_pct` (61.8% retracement + buffer, clamped to existing ATR bounds), `compute_fib_take_profit_pcts` (127.2%/161.8% extensions), and `fib_entry_signal` (small nudge into `compute_opportunity_score`). Both spot (`execute.py`) and futures (`execute_futures.py`) call sites switch to a Fib -> ATR -> flat fallback chain; the existing fixed-SL drawdown-recovery override still takes precedence over everything.

**Tech Stack:** Python 3.9, `decimal.Decimal` throughout (no floats in trading math), existing `requests`-based Binance kline fetch pattern, existing `_test_*` / `if __name__ == "__main__"` self-check convention (no pytest in this repo).

**Reference spec:** `docs/superpowers/specs/2026-08-01-fibonacci-prediction-mechanism-design.md`

## Global Constraints

- No pytest — this repo's test convention is `_test_*()` functions run from `if __name__ == "__main__"` blocks, asserting directly. Follow it exactly; do not introduce a test framework.
- All price/percentage math uses `Decimal`, never `float`, except inside `scanner.py`'s pure candle-math helpers which already operate on `float` (matching `_compute_atr`'s existing convention — the boundary from `float` to `Decimal` happens at the `execute.py` call sites, not inside `scanner.py`).
- New env-configurable constants follow the existing pattern: `NAME = Decimal(os.environ.get("NAME", "default"))` at module scope, documented in `.env.example` in the same style as the neighboring `ATR_*` block.
- Every new function that can fail (network, insufficient data, degenerate input) returns `None` rather than raising — matches `get_atr`/`compute_atr_stop_loss_pct`'s existing failure contract. Callers already know how to treat `None` as "fall back."
- Run the full existing `if __name__ == "__main__"` test block in `execute.py` (`venv/bin/python -c "import execute; execute._test_...(); ..."`) after every task — regressions in unrelated tests block moving to the next task.

---

### Task 1: `get_fibonacci_levels` in scanner.py

**Files:**
- Modify: `scanner.py` (add near `_compute_atr`/`get_atr`, which currently sit around line 288-326; insert the new functions directly after `_test_compute_atr` at line ~333)

**Interfaces:**
- Produces: `_compute_fibonacci_levels(candles: list[tuple[float, float, float]], is_short: bool = False) -> dict | None` — pure function, candles are `(high, low, close)` tuples oldest-first (same shape `_compute_atr` takes).
- Produces: `get_fibonacci_levels(symbol: str, is_short: bool = False, interval: str = "1h", period: int = 14) -> dict | None` — fetch wrapper.
- Return shape on success: `{"swing_high": float, "swing_low": float, "retracements": {23.6: float, 38.2: float, 50.0: float, 61.8: float, 78.6: float}, "extensions": {127.2: float, 161.8: float}}`.

- [ ] **Step 1: Write the failing tests**

Add directly after `_test_compute_atr` (currently ends at line 333, right before `_compute_vwap`) in `scanner.py`:

```python
def _test_compute_fibonacci_levels() -> None:
    # long: swing from 100 (low) to 110 (high), span 10
    candles = [(110.0, 100.0, 105.0) for _ in range(15)]
    levels = _compute_fibonacci_levels(candles, is_short=False)
    assert levels is not None
    assert levels["swing_high"] == 110.0 and levels["swing_low"] == 100.0
    # retracement measured DOWN from the high for a long
    assert abs(levels["retracements"][61.8] - 103.82) < 1e-9, levels["retracements"][61.8]
    assert abs(levels["retracements"][50.0] - 105.0) < 1e-9
    # extension measured UP beyond the high for a long
    assert abs(levels["extensions"][127.2] - 112.72) < 1e-9, levels["extensions"][127.2]
    assert abs(levels["extensions"][161.8] - 116.18) < 1e-9

    # short: same swing, mirrored — retracement UP from the low, extension DOWN beyond the low
    levels_short = _compute_fibonacci_levels(candles, is_short=True)
    assert abs(levels_short["retracements"][61.8] - 106.18) < 1e-9, levels_short["retracements"][61.8]
    assert abs(levels_short["extensions"][127.2] - 97.28) < 1e-9, levels_short["extensions"][127.2]

    # degenerate swing (flat price action, high == low) -> None
    flat = [(100.0, 100.0, 100.0) for _ in range(15)]
    assert _compute_fibonacci_levels(flat) is None

    # thin history -> None
    assert _compute_fibonacci_levels([(110.0, 100.0, 105.0)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from scanner import _test_compute_fibonacci_levels; _test_compute_fibonacci_levels()"`
Expected: `ImportError: cannot import name '_compute_fibonacci_levels'`

- [ ] **Step 3: Write the minimal implementation**

Insert directly after `_test_compute_atr` (line 333) and before `_compute_vwap` in `scanner.py`:

```python
def _compute_fibonacci_levels(candles: list[tuple[float, float, float]], is_short: bool = False) -> dict | None:
    """Fibonacci retracement + extension levels over `candles` — each a
    (high, low, close) tuple, oldest first, same shape _compute_atr takes.
    Pure function; get_fibonacci_levels() wraps it with the actual candle
    fetch.

    For a long (is_short=False): swing measured retracing DOWN from the
    window's high toward its low (support levels below current price);
    extensions measured UP beyond the high (profit targets above price).
    For a short (is_short=True): mirrored — retracing UP from the low
    toward the high (resistance above price); extensions DOWN beyond the
    low (profit targets below price).

    Returns None if there's less than a full period of candles, or the
    swing is degenerate (high <= low, e.g. completely flat price action —
    guards the division these levels feed downstream)."""
    if len(candles) < 14:
        return None
    highs = [h for h, _, _ in candles]
    lows = [l for _, l, _ in candles]
    swing_high, swing_low = max(highs), min(lows)
    if swing_high <= swing_low:
        return None

    span = swing_high - swing_low
    retracement_ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    extension_ratios = (1.272, 1.618)

    if is_short:
        retracements = {round(r * 100, 1): swing_low + span * r for r in retracement_ratios}
        extensions = {round(r * 100, 1): swing_low - span * (r - 1) for r in extension_ratios}
    else:
        retracements = {round(r * 100, 1): swing_high - span * r for r in retracement_ratios}
        extensions = {round(r * 100, 1): swing_high + span * (r - 1) for r in extension_ratios}

    return {"swing_high": swing_high, "swing_low": swing_low, "retracements": retracements, "extensions": extensions}


def get_fibonacci_levels(symbol: str, is_short: bool = False, interval: str = "1h", period: int = 14) -> dict | None:
    """Fibonacci levels over the last `period` closed candles — same
    fetch/compute split as get_atr, same default window (14x1h) so this
    reuses the timeframe concept ATR already established rather than
    introducing a second one. None on a fetch failure or degenerate swing;
    caller falls back to the ATR-based stop/TP."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": f"{symbol}USDT", "interval": interval, "limit": period},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    candles = [(float(k[2]), float(k[3]), float(k[4])) for k in resp.json()]  # high, low, close
    return _compute_fibonacci_levels(candles, is_short=is_short)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from scanner import _test_compute_fibonacci_levels; _test_compute_fibonacci_levels(); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Wire the test into scanner.py's main test block and run the full scanner.py suite**

In `scanner.py`, find the `if __name__ == "__main__":` block (currently starts at line 867) and add `_test_compute_fibonacci_levels()` directly after the existing `_test_compute_atr()` call:

```python
    _test_compute_atr()
    _test_compute_fibonacci_levels()
    _test_compute_vwap()
```

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
import scanner
scanner._test_rank_symbols()
scanner._test_equal_profit_fractions()
scanner._test_pump_signature()
scanner._test_compute_rsi()
scanner._test_compute_atr()
scanner._test_compute_fibonacci_levels()
scanner._test_compute_vwap()
scanner._test_compute_obv()
scanner._test_detect_obv_divergence()
scanner._test_get_text_sentiment()
scanner._test_get_market_breadth()
scanner._test_describe_events()
scanner._test_describe_pump_risk()
print('ALL SCANNER TESTS PASS')
"`
Expected: `ALL SCANNER TESTS PASS`, no traceback.

- [ ] **Step 6: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add scanner.py
git commit -m "$(cat <<'EOF'
Add get_fibonacci_levels: retracement/extension price levels

Pure-function/fetch split matching get_atr's existing convention,
same 14x1h candle window. Feeds the SL/TP/entry-signal work in the
next tasks — this task only adds the level computation itself.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fib-based stop-loss and take-profit in execute.py

**Files:**
- Modify: `execute.py` (add near `ATR_MIN_STOP_LOSS_PCT`/`compute_atr_stop_loss_pct`, currently lines 417-460ish)

**Interfaces:**
- Consumes: `get_fibonacci_levels(symbol, is_short, interval, period)` from Task 1 (must be imported from `scanner` into `execute.py`).
- Produces: `compute_fib_stop_loss_pct(symbol: str, price: Decimal, is_short: bool = False) -> Decimal | None`
- Produces: `compute_fib_take_profit_pcts(symbol: str, price: Decimal, is_short: bool = False) -> tuple[Decimal, Decimal] | None`
- Produces: module constant `FIB_STOP_BUFFER_PCT`

- [ ] **Step 1: Add the scanner import**

In `execute.py`, the `from scanner import (...)` block (lines 18-34) currently reads:

```python
from scanner import (
    MEME_SYMBOLS,
    equal_profit_fractions,
    get_atr,
    get_binance_momentum_short,
    get_binance_universe,
    get_cmc_movers,
    get_dexscreener_trend,
    get_fear_greed_index,
    get_google_trends,
    get_kronos_forecast,
    get_llm_verdict,
    get_market_breadth,
    get_obv_divergence,
    get_rsi,
    get_vwap,
    rank_symbols,
)
```

Add `get_fibonacci_levels,` alphabetically (between `get_fear_greed_index` and `get_google_trends`):

```python
from scanner import (
    MEME_SYMBOLS,
    equal_profit_fractions,
    get_atr,
    get_binance_momentum_short,
    get_binance_universe,
    get_cmc_movers,
    get_dexscreener_trend,
    get_fear_greed_index,
    get_fibonacci_levels,
    get_google_trends,
    get_kronos_forecast,
    get_llm_verdict,
    get_market_breadth,
    get_obv_divergence,
    get_rsi,
    get_vwap,
    rank_symbols,
)
```

- [ ] **Step 2: Write the failing tests**

Find `_test_compute_atr_stop_loss_pct` in `execute.py` (currently line 440) and add directly after it:

```python
def _test_compute_fib_stop_loss_pct() -> None:
    # bad symbol -> get_fibonacci_levels can't fetch -> None -> caller falls back to ATR
    assert compute_fib_stop_loss_pct("__NOPE__", Decimal("100")) is None
    # price <= 0 -> also None rather than dividing by zero
    assert compute_fib_stop_loss_pct("BTC", Decimal("0")) is None


def _test_compute_fib_take_profit_pcts() -> None:
    assert compute_fib_take_profit_pcts("__NOPE__", Decimal("100")) is None
    assert compute_fib_take_profit_pcts("BTC", Decimal("0")) is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from execute import _test_compute_fib_stop_loss_pct; _test_compute_fib_stop_loss_pct()"`
Expected: `NameError: name 'compute_fib_stop_loss_pct' is not defined`

- [ ] **Step 4: Write the minimal implementation**

In `execute.py`, find `DEFAULT_STOP_LOSS_PCT = Decimal("0.15")  # fallback when ATR is unavailable this cycle` (currently line 420) and add directly after it, before `def compute_atr_stop_loss_pct`:

```python
FIB_STOP_BUFFER_PCT = Decimal(os.environ.get("FIB_STOP_BUFFER_PCT", "0.02"))


def compute_fib_stop_loss_pct(symbol: str, price: Decimal, is_short: bool = False) -> Decimal | None:
    """Stop-loss sized to the 61.8% Fibonacci retracement level (the
    "golden ratio" invalidation level) instead of pure ATR volatility —
    ties the stop to actual price structure. Pushed FIB_STOP_BUFFER_PCT
    further out so the stop doesn't sit exactly on the level Fib traders
    themselves watch (mirrors place_oco_exit's stop_price * 0.995 "hair
    below the trigger" idea). Clamped to the same [ATR_MIN_STOP_LOSS_PCT,
    ATR_MAX_STOP_LOSS_PCT] bounds ATR uses, so a distant swing can't
    produce an outlier stop either. Returns None if Fibonacci levels
    aren't available this cycle — caller falls back to
    compute_atr_stop_loss_pct."""
    if price <= 0:
        return None
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return None
    level_61_8 = Decimal(str(levels["retracements"][61.8]))
    raw_pct = abs(price - level_61_8) / price
    buffered_pct = raw_pct * (1 + FIB_STOP_BUFFER_PCT)
    return max(ATR_MIN_STOP_LOSS_PCT, min(ATR_MAX_STOP_LOSS_PCT, buffered_pct))


def compute_fib_take_profit_pcts(symbol: str, price: Decimal, is_short: bool = False) -> tuple[Decimal, Decimal] | None:
    """Take-profit legs sized to the 127.2% and 161.8% Fibonacci extension
    levels instead of a flat multiple of the stop distance (ATR_TP_RATIO_1/2)
    — independent of whatever stop_loss_pct ends up chosen. Returns None if
    Fibonacci levels aren't available this cycle — caller falls back to
    compute_atr_take_profit_pcts."""
    if price <= 0:
        return None
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return None
    tp1 = Decimal(str(levels["extensions"][127.2]))
    tp2 = Decimal(str(levels["extensions"][161.8]))
    return (abs(tp1 - price) / price, abs(tp2 - price) / price)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
from execute import _test_compute_fib_stop_loss_pct, _test_compute_fib_take_profit_pcts
_test_compute_fib_stop_loss_pct()
_test_compute_fib_take_profit_pcts()
print('ok')
"`
Expected: `ok` (a `NotOpenSSLWarning` line from urllib3 is expected noise, not a failure)

- [ ] **Step 6: Wire into execute.py's main test block**

Find `if __name__ == "__main__":` in `execute.py` (currently line 1432) and add the two new calls directly after `_test_compute_atr_stop_loss_pct()`:

```python
    _test_compute_atr_stop_loss_pct()
    _test_compute_fib_stop_loss_pct()
    _test_compute_fib_take_profit_pcts()
    _test_use_fixed_stop_loss()
```

- [ ] **Step 7: Run the full execute.py test suite**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
import execute, tempfile, os
execute._test_round_step()
execute._test_should_exit_early()
execute._test_update_peak_and_drawdown(os.path.join(tempfile.gettempdir(), '_test_peak.json'))
execute._test_confirm_momentum(os.path.join(tempfile.gettempdir(), '_test_confirm.json'))
execute._test_golden_trade_marking(os.path.join(tempfile.gettempdir(), '_test_golden.json'))
execute._test_compute_opportunity_score()
execute._test_compute_atr_stop_loss_pct()
execute._test_compute_fib_stop_loss_pct()
execute._test_compute_fib_take_profit_pcts()
execute._test_use_fixed_stop_loss()
execute._test_meets_min_reward_risk()
execute._test_check_daily_loss_limit(os.path.join(tempfile.gettempdir(), '_test_daily_loss.json'))
execute._test_compute_atr_take_profit_pcts()
execute._test_is_counter_trend()
execute._test_vwap_deviation()
execute._test_obv_warns_against()
print('ALL EXECUTE TESTS PASS')
"`
Expected: `ALL EXECUTE TESTS PASS`, no traceback.

- [ ] **Step 8: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute.py
git commit -m "$(cat <<'EOF'
Add compute_fib_stop_loss_pct/compute_fib_take_profit_pcts

Stop-loss anchors to the 61.8% Fibonacci retracement (plus a small
buffer), clamped to the existing ATR bounds. Take-profit anchors to
the 127.2%/161.8% extensions, independent of whatever stop got
chosen. Not wired into the trading loop yet — that's the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fib entry signal + `fib_score` on `compute_opportunity_score`

**Files:**
- Modify: `execute.py`

**Interfaces:**
- Consumes: `get_fibonacci_levels` (already imported in Task 2), `compute_opportunity_score` (existing, being extended).
- Produces: `fib_entry_signal(symbol: str, price: Decimal, is_short: bool = False) -> Decimal` (never `None` — returns `Decimal("0")` on missing data).
- Modifies: `compute_opportunity_score` gains a new keyword-only-by-convention param `fib_score: Decimal = Decimal("0")`.

- [ ] **Step 1: Write the failing tests**

Find `_test_compute_fib_take_profit_pcts` (added in Task 2) in `execute.py` and add directly after it:

```python
def _test_fib_entry_signal() -> None:
    assert fib_entry_signal("__NOPE__", Decimal("100")) == Decimal("0")
    assert fib_entry_signal("BTC", Decimal("0")) == Decimal("0")
```

Find `_test_compute_opportunity_score` (currently line 797) and add these lines directly before the final `print("compute_opportunity_score self-check OK")`:

```python
    # fib structure nudge
    base_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"))["opportunity_score"]
    bullish_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("0.1"))
    assert bullish_fib["opportunity_score"] > base_fib, "positive fib_score should raise the opportunity score"
    bearish_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("-0.1"))
    assert bearish_fib["opportunity_score"] < base_fib, "negative fib_score should lower the opportunity score"
    neutral_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("0"))
    assert neutral_fib["opportunity_score"] == base_fib, "zero fib_score (the default) should be a no-op"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from execute import _test_fib_entry_signal; _test_fib_entry_signal()"`
Expected: `NameError: name 'fib_entry_signal' is not defined`

- [ ] **Step 3: Write the minimal implementation**

In `execute.py`, add directly after `compute_fib_take_profit_pcts` (added in Task 2, right before `_test_compute_fib_stop_loss_pct`... actually before whichever test block follows — insert the function above its own test, i.e. directly after `compute_fib_take_profit_pcts`'s closing line and before `_test_compute_fib_stop_loss_pct`):

```python
FIB_SIGNAL_NUDGE = Decimal(os.environ.get("FIB_SIGNAL_NUDGE", "0.1"))


def fib_entry_signal(symbol: str, price: Decimal, is_short: bool = False) -> Decimal:
    """Small additive nudge into compute_opportunity_score's fib_score —
    same pattern and magnitude as the Fear & Greed nudge. Positive if price
    is holding on the "structurally intact" side of the 50% retracement
    (above it for a long, below it for a short); negative if price has
    already broken past the 61.8% level in the wrong direction (structure
    weakening, thesis in doubt). Neutral (0) if Fibonacci levels aren't
    available this cycle — never blocks, mirrors how fear_greed=None means
    no nudge in compute_opportunity_score."""
    if price <= 0:
        return Decimal("0")
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return Decimal("0")
    level_50 = Decimal(str(levels["retracements"][50.0]))
    level_61_8 = Decimal(str(levels["retracements"][61.8]))
    if is_short:
        if price <= level_50:
            return FIB_SIGNAL_NUDGE
        if price >= level_61_8:
            return -FIB_SIGNAL_NUDGE
    else:
        if price >= level_50:
            return FIB_SIGNAL_NUDGE
        if price <= level_61_8:
            return -FIB_SIGNAL_NUDGE
    return Decimal("0")
```

Then modify `compute_opportunity_score`'s signature (currently lines 719-722):

```python
def compute_opportunity_score(
    llm_confidence: Decimal, kronos_pct: Decimal | None,
    fear_greed: int | None = None, is_short: bool = False,
) -> dict:
```

to:

```python
def compute_opportunity_score(
    llm_confidence: Decimal, kronos_pct: Decimal | None,
    fear_greed: int | None = None, is_short: bool = False, fib_score: Decimal = Decimal("0"),
) -> dict:
```

Add one bullet to the docstring (after the `is_short` bullet, before the closing `"""` — the docstring currently ends `"...bullish enough to risk a squeeze")."""` around line 750):

```
    - fib_score: additive nudge from fib_entry_signal (see that function) —
      structural confirmation from Fibonacci retracement levels, on the
      same footing as the Fear & Greed nudge. Defaults to 0 (no effect) so
      every existing caller keeps working unchanged.
    """
```

Change the score-blend line (currently `opportunity_score = KRONOS_WEIGHT * kronos_score + LLM_WEIGHT * llm_score` at line 758) to:

```python
    opportunity_score = KRONOS_WEIGHT * kronos_score + LLM_WEIGHT * llm_score + fib_score
```

Add a reasoning-string fragment. Currently (around lines 781-789):

```python
    reasoning = (
        f"Kronos: {kronos_score:+.2f} conviction ({direction}"
        + (f", predicts {kronos_pct:+.2f}%" if kronos_pct is not None else ", no forecast this cycle")
        + f") x weight {KRONOS_WEIGHT} = {KRONOS_WEIGHT * kronos_score:+.2f}. "
        f"LLM: {llm_score:.2f} confidence x weight {LLM_WEIGHT} = {LLM_WEIGHT * llm_score:+.2f}."
        f"{fear_greed_note} "
        f"Opportunity score {opportunity_score:+.2f} vs threshold {OPPORTUNITY_THRESHOLD} -> "
        + ("PASSES." if passes else f"FAILS" + (f" (Kronos veto — {veto_reason} regardless of LLM confidence)." if veto else "."))
    )
```

Change to (adds one `fib_note` fragment computed just above the `reasoning` assignment, and splices it in next to `fear_greed_note`):

```python
    fib_note = f" Fib structure nudge {fib_score:+.2f}." if fib_score != 0 else ""
    reasoning = (
        f"Kronos: {kronos_score:+.2f} conviction ({direction}"
        + (f", predicts {kronos_pct:+.2f}%" if kronos_pct is not None else ", no forecast this cycle")
        + f") x weight {KRONOS_WEIGHT} = {KRONOS_WEIGHT * kronos_score:+.2f}. "
        f"LLM: {llm_score:.2f} confidence x weight {LLM_WEIGHT} = {LLM_WEIGHT * llm_score:+.2f}."
        f"{fear_greed_note}{fib_note} "
        f"Opportunity score {opportunity_score:+.2f} vs threshold {OPPORTUNITY_THRESHOLD} -> "
        + ("PASSES." if passes else f"FAILS" + (f" (Kronos veto — {veto_reason} regardless of LLM confidence)." if veto else "."))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
from execute import _test_fib_entry_signal, _test_compute_opportunity_score
_test_fib_entry_signal()
_test_compute_opportunity_score()
print('ok')
"`
Expected: `compute_opportunity_score self-check OK` followed by `ok`.

- [ ] **Step 5: Wire into execute.py's main test block**

In the `if __name__ == "__main__":` block, add `_test_fib_entry_signal()` directly after `_test_compute_fib_take_profit_pcts()`:

```python
    _test_compute_fib_stop_loss_pct()
    _test_compute_fib_take_profit_pcts()
    _test_fib_entry_signal()
    _test_use_fixed_stop_loss()
```

- [ ] **Step 6: Run the full execute.py test suite**

Run the same full command as Task 2 Step 7, plus confirm `_test_fib_entry_signal` and the updated `_test_compute_opportunity_score` both run clean:

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
import execute, tempfile, os
execute._test_round_step()
execute._test_should_exit_early()
execute._test_update_peak_and_drawdown(os.path.join(tempfile.gettempdir(), '_test_peak.json'))
execute._test_confirm_momentum(os.path.join(tempfile.gettempdir(), '_test_confirm.json'))
execute._test_golden_trade_marking(os.path.join(tempfile.gettempdir(), '_test_golden.json'))
execute._test_compute_opportunity_score()
execute._test_compute_atr_stop_loss_pct()
execute._test_compute_fib_stop_loss_pct()
execute._test_compute_fib_take_profit_pcts()
execute._test_fib_entry_signal()
execute._test_use_fixed_stop_loss()
execute._test_meets_min_reward_risk()
execute._test_check_daily_loss_limit(os.path.join(tempfile.gettempdir(), '_test_daily_loss.json'))
execute._test_compute_atr_take_profit_pcts()
execute._test_is_counter_trend()
execute._test_vwap_deviation()
execute._test_obv_warns_against()
print('ALL EXECUTE TESTS PASS')
"
```
Expected: `ALL EXECUTE TESTS PASS`, no traceback.

- [ ] **Step 7: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute.py
git commit -m "$(cat <<'EOF'
Add fib_entry_signal, wire fib_score into compute_opportunity_score

Small additive nudge (same pattern/magnitude as the Fear & Greed
nudge) based on whether price is holding the structurally-intact
side of the 50%/61.8% Fibonacci retracement. Defaults to 0 so every
existing caller of compute_opportunity_score is unaffected until the
next task wires it in at the actual call sites.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire the fallback chain into execute.py's spot trading loop

**Files:**
- Modify: `execute.py` (the per-candidate loop inside the spot cycle function, currently lines ~1287-1329)

**Interfaces:**
- Consumes: `fib_entry_signal`, `compute_fib_stop_loss_pct`, `compute_fib_take_profit_pcts` (Tasks 2-3), `compute_opportunity_score`'s new `fib_score` param (Task 3).
- No new functions produced — this task only rewires an existing call site. Verified by the full regression suite (no new unit test possible without mocking the live exchange client, which is out of scope — matches how the existing `use_fixed_stop_loss()`/`compute_atr_stop_loss_pct()` wiring at this same call site has never had a dedicated call-site test either).

- [ ] **Step 1: Make the change**

In `execute.py`, find this block (currently lines 1317-1326):

```python
        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        opportunity = compute_opportunity_score(Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed)
        print(f"  Opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the Kronos+LLM blended gate.")
            continue

        stop_loss_pct = DEFAULT_STOP_LOSS_PCT if use_fixed_stop_loss() else compute_atr_stop_loss_pct(sym, current_price)
        tp_pcts = compute_atr_take_profit_pcts(stop_loss_pct)
        if not meets_min_reward_risk(tp_pcts[0], stop_loss_pct):
```

Replace with:

```python
        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        fib_score = fib_entry_signal(sym, current_price)
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, fib_score=fib_score,
        )
        print(f"  Opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the Kronos+LLM blended gate.")
            continue

        if use_fixed_stop_loss():
            stop_loss_pct = DEFAULT_STOP_LOSS_PCT
        else:
            stop_loss_pct = compute_fib_stop_loss_pct(sym, current_price) or compute_atr_stop_loss_pct(sym, current_price)
        tp_pcts = compute_fib_take_profit_pcts(sym, current_price) or compute_atr_take_profit_pcts(stop_loss_pct)
        if not meets_min_reward_risk(tp_pcts[0], stop_loss_pct):
```

- [ ] **Step 2: Confirm the module still imports cleanly**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "import execute; print('import ok')"`
Expected: `import ok`, no `NameError`/`SyntaxError`.

- [ ] **Step 3: Run the full execute.py regression suite**

Run the exact same command as Task 3 Step 6.
Expected: `ALL EXECUTE TESTS PASS`, no traceback. This confirms the rewired call site didn't break any of the underlying functions it composes — the functions themselves already have direct unit coverage from Tasks 1-3.

- [ ] **Step 4: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute.py
git commit -m "$(cat <<'EOF'
Wire Fib -> ATR -> flat fallback chain into the spot trading loop

New trades now try compute_fib_stop_loss_pct/compute_fib_take_profit_pcts
first, falling back to the existing ATR-based functions when
Fibonacci levels aren't available. fib_entry_signal now feeds
compute_opportunity_score alongside Kronos+LLM. The fixed-SL
drawdown-recovery policy (use_fixed_stop_loss) still takes precedence
over both when active — unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire the fallback chain into execute_futures.py's short trading loop

**Files:**
- Modify: `execute_futures.py` (the `from execute import (...)` block, currently lines 27-41; the per-candidate loop, currently lines ~384-404)

**Interfaces:**
- Consumes: `fib_entry_signal`, `compute_fib_stop_loss_pct`, `compute_fib_take_profit_pcts` from `execute.py` (Tasks 2-3) — must be added to the existing import block.

- [ ] **Step 1: Add the imports**

In `execute_futures.py`, the `from execute import (...)` block currently reads:

```python
from execute import (
    DAILY_LOSS_LIMIT_PCT,
    MIN_REWARD_RISK_RATIO,
    _record_traded_symbol,
    _round_step,
    check_daily_loss_limit,
    compute_atr_stop_loss_pct,
    compute_atr_take_profit_pcts,
    compute_vwap_deviation_pct,
    is_counter_trend,
    is_extended_from_vwap,
    kill_switch_active,
    use_fixed_stop_loss,
    make_trade_id,
    meets_min_reward_risk,
    obv_warns_against,
)
```

Add `compute_fib_stop_loss_pct`, `compute_fib_take_profit_pcts`, and `fib_entry_signal` alphabetically:

```python
from execute import (
    DAILY_LOSS_LIMIT_PCT,
    MIN_REWARD_RISK_RATIO,
    _record_traded_symbol,
    _round_step,
    check_daily_loss_limit,
    compute_atr_stop_loss_pct,
    compute_atr_take_profit_pcts,
    compute_fib_stop_loss_pct,
    compute_fib_take_profit_pcts,
    compute_vwap_deviation_pct,
    fib_entry_signal,
    is_counter_trend,
    is_extended_from_vwap,
    kill_switch_active,
    use_fixed_stop_loss,
    make_trade_id,
    meets_min_reward_risk,
    obv_warns_against,
)
```

- [ ] **Step 2: Rewire the call site**

Find this block (currently lines 394-404):

```python
        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, is_short=True,
        )
        print(f"  Short opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the gate.")
            continue

        stop_loss_pct = FUTURES_STOP_LOSS_PCT if use_fixed_stop_loss() else compute_atr_stop_loss_pct(sym, current_price)
        take_profit_pct = compute_atr_take_profit_pcts(stop_loss_pct)[0]  # shorts aren't laddered, leg 1 only
```

Replace with:

```python
        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        fib_score = fib_entry_signal(sym, current_price, is_short=True)
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, is_short=True, fib_score=fib_score,
        )
        print(f"  Short opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the gate.")
            continue

        if use_fixed_stop_loss():
            stop_loss_pct = FUTURES_STOP_LOSS_PCT
        else:
            stop_loss_pct = compute_fib_stop_loss_pct(sym, current_price, is_short=True) or compute_atr_stop_loss_pct(sym, current_price)
        fib_tp = compute_fib_take_profit_pcts(sym, current_price, is_short=True)
        take_profit_pct = fib_tp[0] if fib_tp else compute_atr_take_profit_pcts(stop_loss_pct)[0]  # shorts aren't laddered, leg 1 only
```

- [ ] **Step 3: Confirm both modules still import cleanly (checks for circular imports too)**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "import execute_futures; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 4: Re-run the full execute.py regression suite**

Run the exact same command as Task 3 Step 6 (execute_futures.py has no independent `_test_*` suite of its own beyond `_test_should_exit_short_early`/`_test_confirm_worst_candidate`, which are unaffected by this change — the functions being wired in are already covered by execute.py's suite).
Expected: `ALL EXECUTE TESTS PASS`, no traceback.

- [ ] **Step 5: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute_futures.py
git commit -m "$(cat <<'EOF'
Wire Fib -> ATR -> flat fallback chain into the futures short loop

Mirrors the spot-side wiring from the previous commit. New shorts
now try compute_fib_stop_loss_pct/compute_fib_take_profit_pcts
first, falling back to ATR-based when Fibonacci levels aren't
available. fib_entry_signal feeds compute_opportunity_score
alongside Kronos+LLM on the short side too. use_fixed_stop_loss
still takes precedence over both when the drawdown-recovery policy
is active.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Document the new env vars and restart the live loop

**Files:**
- Modify: `.env.example`
- Restart: `loop.py` (running process must reload to pick up the code change — same requirement as the fixed-SL policy change earlier this session)

**Interfaces:** None — documentation and operational step only.

- [ ] **Step 1: Add documentation to `.env.example`**

Find the existing ATR block (currently lines 123-131):

```
# ATR-based (volatility-adjusted) stop-loss, replacing one flat % for every
# symbol on BOTH the spot-long and futures-short paths - Turtle-style:
# stop = ATR_MULTIPLIER x ATR(14, 1h), expressed as % of price, clamped to
# [ATR_MIN_STOP_LOSS_PCT, ATR_MAX_STOP_LOSS_PCT]. Falls back to the flat 15%
# default (DEFAULT_STOP_LOSS_PCT / FUTURES_STOP_LOSS_PCT) if ATR can't be
# computed this cycle.
ATR_MULTIPLIER=2
ATR_MIN_STOP_LOSS_PCT=0.08
ATR_MAX_STOP_LOSS_PCT=0.25
```

Add directly after it:

```
# Fibonacci-anchored stop-loss/take-profit/entry-signal, tried BEFORE the
# ATR block above on both the spot-long and futures-short paths (Fib -> ATR
# -> flat 15% default fallback chain). Stop-loss sits FIB_STOP_BUFFER_PCT
# beyond the 61.8% retracement level (clamped to the same
# [ATR_MIN_STOP_LOSS_PCT, ATR_MAX_STOP_LOSS_PCT] bounds above); take-profit
# targets the 127.2%/161.8% extension levels; FIB_SIGNAL_NUDGE is the
# entry-side nudge into the Kronos+LLM opportunity score, same pattern and
# magnitude as FEAR_GREED_NUDGE below. Falls back to the ATR block above if
# Fibonacci levels can't be computed this cycle (thin history, degenerate
# swing, or a fetch failure).
FIB_STOP_BUFFER_PCT=0.02
FIB_SIGNAL_NUDGE=0.1
```

- [ ] **Step 2: Commit the documentation**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add .env.example
git commit -m "$(cat <<'EOF'
Document FIB_STOP_BUFFER_PCT/FIB_SIGNAL_NUDGE in .env.example

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Restart the live loop to pick up the code change**

`loop.py` is a long-running background process — it has the pre-Fibonacci code loaded in memory and will not pick up any of Tasks 1-5 until restarted (same requirement as the fixed-SL policy change earlier this session).

```bash
ps aux | grep loop.py | grep -v grep
```

Note the PID, then:

```bash
kill <PID>
sleep 2
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
set -a && source .env && set +a && nohup ./venv/bin/python -u loop.py >> loop.log 2>&1 &
disown
sleep 3
ps aux | grep loop.py | grep -v grep
```

- [ ] **Step 4: Verify the new code path is live**

```bash
sleep 5
tail -20 loop.log
```

Confirm the process is running under a new PID and producing fresh timestamped cycle output (`=== <timestamp> ===`), same verification pattern used after the fixed-SL restart earlier this session. Note: the "ATR-based stop-loss: X% (vs flat Y% default)" print lines at the call sites were not changed by this plan and will still print even when the value actually came from Fib — cosmetic only, not a correctness issue, and out of scope per the spec's "out of scope" section.

---

## Self-Review Notes

- **Spec coverage:** every section of the spec (`get_fibonacci_levels`, `compute_fib_stop_loss_pct`, `compute_fib_take_profit_pcts`, `fib_entry_signal`, both call-site rewires, fixed-SL precedence, `FIB_STOP_BUFFER_PCT`, testing convention) maps to a task above. The spec's "out of scope" items (retiring ATR functions, changing fixed-SL's own logic, a separate swing window, Fib-based ladder-quantity split) are untouched by every task.
- **Type consistency checked:** `get_fibonacci_levels` returns `float` values inside its dict (matching `_compute_atr`'s float convention); every `execute.py` consumer converts via `Decimal(str(...))` before doing `Decimal` math, same boundary-conversion pattern `compute_atr_stop_loss_pct` already uses at the `scanner.py`/`execute.py` boundary. `is_short` is threaded consistently as a keyword arg through every new function, matching every existing signal function's convention (`is_counter_trend`, `is_extended_from_vwap`, `obv_warns_against`).
- **No placeholders:** every step has runnable code; no "TBD"/"similar to Task N" shortcuts.
