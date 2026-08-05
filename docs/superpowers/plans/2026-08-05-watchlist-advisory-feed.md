# Watchlist Advisory Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the bot a per-symbol memory that biases entry decisions — a decaying cooldown penalty after stop-outs (scaled by how often that symbol has burned us) and a small bonus for symbols repeatedly rejected only for lack of capital — without ever forcing an entry.

**Architecture:** One new self-contained module `watchlist.py` holding two pure nudge functions and a small JSON state layer, consumed by both books through per-book state files. Three additive integration hooks: a defaulted `watchlist_nudge` parameter on `compute_opportunity_score` (identical pattern to the existing `fear_greed`/`fib_score` nudges), stop-out recording in both journal classifiers, and capital-block recording at the four post-`chosen` rejection paths.

**Tech Stack:** Python 3.9, `decimal.Decimal` for all score math, `datetime` with timezone-aware UTC, plain JSON files for state. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-08-05-watchlist-advisory-feed-design.md`

## Global Constraints

- **No pytest.** This repo's convention is `_test_*()` functions containing bare `assert`s, invoked from an `if __name__ == "__main__":` block. Follow it exactly; do not introduce a test framework.
- **All score math uses `Decimal`, never `float`.** Timestamps are stored as timezone-aware UTC ISO strings via `datetime.now(timezone.utc).isoformat()`.
- **New constants are env-overridable** at module scope: `NAME = Decimal(os.environ.get("NAME", "default"))`, matching every existing constant in `execute.py`.
- **Nothing may raise into a trading cycle.** The watchlist is advisory — every failure path degrades to "no nudge" / empty state rather than propagating. A missing or empty state file is normal, not an error. Malformed timestamps are skipped, not fatal.
- **Per-book state files are mandatory, not stylistic.** A *short* stop-out means price rose, which is bullish information for a *long*; a shared file would leak the inverted signal across books. Same reasoning already applied to `trade_peaks.json` / `trade_peaks_futures.json`.
- Run the full regression suite after every task (command given in Task 5); a regression in an unrelated test blocks moving on.

---

### Task 1: Pure nudge functions in `watchlist.py`

**Files:**
- Create: `watchlist.py`

**Interfaces:**
- Produces: `cooldown_penalty(stop_out_times: list[str], now: datetime) -> Decimal` — returns ≤ 0
- Produces: `capital_block_bonus(block_times: list[str], now: datetime) -> Decimal` — returns ≥ 0
- Produces: `_parse_times(times: list[str]) -> list[datetime]` — skips malformed entries
- Produces: constants `WATCHLIST_COOLDOWN_BASE`, `WATCHLIST_COOLDOWN_HOURS`, `WATCHLIST_CAPITAL_BONUS`, `WATCHLIST_MIN_BLOCKS`, `WATCHLIST_MEMORY_HOURS`

- [ ] **Step 1: Write the failing test**

Create `watchlist.py` containing only this test function for now:

```python
def _test_nudge_math() -> None:
    now = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)

    def ago(hours):
        return (now - timedelta(hours=hours)).isoformat()

    # --- cooldown_penalty ---
    assert cooldown_penalty([], now) == Decimal("0")  # no history -> no penalty
    # one stop-out just now -> full base penalty, negative
    fresh = cooldown_penalty([ago(0)], now)
    assert abs(fresh - -WATCHLIST_COOLDOWN_BASE) < Decimal("0.001"), fresh
    # halfway through the decay window -> about half strength
    half = cooldown_penalty([ago(float(WATCHLIST_COOLDOWN_HOURS) / 2)], now)
    assert abs(half - -WATCHLIST_COOLDOWN_BASE / 2) < Decimal("0.01"), half
    # fully decayed -> exactly zero, not a tiny negative
    assert cooldown_penalty([ago(float(WATCHLIST_COOLDOWN_HOURS) + 1)], now) == Decimal("0")

    # EPIC regression: twice-burned symbol is penalised about twice as hard.
    # EPIC stopped out at -114.19 then -197.96 (58% of all futures losses) and
    # was re-entered a third time with nothing noting the history.
    once = cooldown_penalty([ago(1)], now)
    twice = cooldown_penalty([ago(1), ago(30)], now)
    assert abs(twice - once * 2) < Decimal("0.01"), (once, twice)
    assert twice < once  # more negative

    # stop-outs older than the memory window stop counting toward the multiplier
    stale = cooldown_penalty([ago(1), ago(float(WATCHLIST_MEMORY_HOURS) + 1)], now)
    assert abs(stale - once) < Decimal("0.001"), stale

    # --- capital_block_bonus ---
    assert capital_block_bonus([], now) == Decimal("0")
    assert capital_block_bonus([ago(1)], now) == Decimal("0")  # 1 block is weak evidence
    assert capital_block_bonus([ago(1), ago(2)], now) == WATCHLIST_CAPITAL_BONUS
    assert capital_block_bonus([ago(1), ago(2), ago(3)], now) == WATCHLIST_CAPITAL_BONUS  # capped, not cumulative
    # blocks outside the memory window don't count toward the threshold
    assert capital_block_bonus([ago(1), ago(float(WATCHLIST_MEMORY_HOURS) + 1)], now) == Decimal("0")

    # --- malformed input degrades, never raises ---
    assert cooldown_penalty(["not-a-timestamp"], now) == Decimal("0")
    assert capital_block_bonus(["not-a-timestamp", ago(1)], now) == Decimal("0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from watchlist import _test_nudge_math; _test_nudge_math()"`
Expected: `NameError: name 'datetime' is not defined` (nothing implemented yet)

- [ ] **Step 3: Write the minimal implementation**

Put this at the top of `watchlist.py`, above the test:

```python
"""Per-symbol memory that BIASES entry decisions without ever forcing one.

Two feeds, both advisory:
  - stop-outs earn a decaying penalty, scaled by how many times that symbol
    has stopped us out. EPIC stopped out twice (-114.19, then -197.96 —
    together 58% of all futures losses) and was re-entered a third time with
    nothing in the logic noting the history.
  - symbols that clear every quality gate and are refused purely for capital
    earn a small bonus once repeatedly blocked. These are the highest-quality
    rejections the bot produces: its own judgment already said yes.

Deliberately advisory only. Nothing here creates a position or bypasses a
gate; the nudge feeds compute_opportunity_score exactly like the existing
Fear & Greed and Fibonacci nudges do.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

WATCHLIST_COOLDOWN_BASE = Decimal(os.environ.get("WATCHLIST_COOLDOWN_BASE", "0.10"))  # penalty per prior stop-out at full strength; matches FEAR_GREED_NUDGE/FIB_SIGNAL_NUDGE against a 0.3 threshold
WATCHLIST_COOLDOWN_HOURS = Decimal(os.environ.get("WATCHLIST_COOLDOWN_HOURS", "24"))  # linear decay back to neutral
WATCHLIST_CAPITAL_BONUS = Decimal(os.environ.get("WATCHLIST_CAPITAL_BONUS", "0.05"))  # half a standard nudge — deliberately weaker than the penalty
WATCHLIST_MIN_BLOCKS = int(os.environ.get("WATCHLIST_MIN_BLOCKS", "2"))  # a single block is coincidence; repetition is signal
WATCHLIST_MEMORY_HOURS = Decimal(os.environ.get("WATCHLIST_MEMORY_HOURS", "168"))  # 7 days: how long records count and are retained


def _parse_times(times: list[str]) -> list[datetime]:
    """ISO strings to datetimes, skipping anything unparseable. The watchlist
    is advisory, so a corrupt record must degrade to "no nudge" rather than
    raise into a live trading cycle."""
    parsed = []
    for t in times:
        try:
            parsed.append(datetime.fromisoformat(t))
        except (ValueError, TypeError):
            continue
    return parsed


def _within_window(times: list[str], now: datetime, hours: Decimal) -> list[datetime]:
    cutoff = now - timedelta(hours=float(hours))
    return [t for t in _parse_times(times) if t >= cutoff]


def cooldown_penalty(stop_out_times: list[str], now: datetime) -> Decimal:
    """Negative nudge making re-entry harder after a stop-out.

    -BASE x offender_count x decay, where offender_count is how many stop-outs
    fall inside WATCHLIST_MEMORY_HOURS and decay falls linearly from 1 to 0
    across WATCHLIST_COOLDOWN_HOURS since the most recent one.

    Scaling by offender_count is the point: it encodes "this symbol
    specifically has burned us repeatedly" rather than treating a third
    attempt like a first."""
    recent = _within_window(stop_out_times, now, WATCHLIST_MEMORY_HOURS)
    if not recent:
        return Decimal("0")
    hours_since = Decimal(str((now - max(recent)).total_seconds() / 3600))
    if hours_since >= WATCHLIST_COOLDOWN_HOURS:
        return Decimal("0")
    decay = 1 - (hours_since / WATCHLIST_COOLDOWN_HOURS)
    return -WATCHLIST_COOLDOWN_BASE * len(recent) * decay


def capital_block_bonus(block_times: list[str], now: datetime) -> Decimal:
    """Positive nudge for a symbol repeatedly refused only for capital.

    Flat once the threshold is met rather than cumulative — the signal is
    "this keeps qualifying", which one extra block doesn't strengthen."""
    recent = _within_window(block_times, now, WATCHLIST_MEMORY_HOURS)
    return WATCHLIST_CAPITAL_BONUS if len(recent) >= WATCHLIST_MIN_BLOCKS else Decimal("0")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from watchlist import _test_nudge_math; _test_nudge_math(); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add watchlist.py
git commit -m "$(cat <<'EOF'
Add watchlist nudge math: stop-out cooldown and capital-block bonus

Pure functions, no I/O. The cooldown scales by how many times a symbol
has stopped us out, so a repeat offender clears a higher bar than a
first-timer — EPIC stopped out twice (-114.19, -197.96, together 58% of
all futures losses) and was re-entered a third time with nothing noting
the history.

State layer and integration follow in later commits.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: JSON state layer in `watchlist.py`

**Files:**
- Modify: `watchlist.py`

**Interfaces:**
- Consumes: `cooldown_penalty`, `capital_block_bonus`, `_within_window`, and the constants from Task 1
- Produces: `WATCHLIST_FILE`, `WATCHLIST_FILE_FUTURES` (module-scope paths)
- Produces: `record_stop_out(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> None`
- Produces: `record_capital_block(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> None`
- Produces: `watchlist_score_nudge(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> Decimal`
- Produces: `prune_watchlist(watchlist_file: str = WATCHLIST_FILE) -> None`
- Produces: `get_watchlist(watchlist_file: str = WATCHLIST_FILE) -> dict`

- [ ] **Step 1: Write the failing test**

Add to `watchlist.py`, after `_test_nudge_math`:

```python
def _test_watchlist_state() -> None:
    import tempfile

    fd, spot_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(spot_file)  # a missing file is normal, not an error
    fd, futures_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(futures_file)
    try:
        # nothing recorded -> empty, zero nudge, no raise
        assert get_watchlist(spot_file) == {}
        assert watchlist_score_nudge("EPIC", spot_file) == Decimal("0")

        # a stop-out produces a negative nudge
        record_stop_out("EPIC", spot_file)
        assert watchlist_score_nudge("EPIC", spot_file) < 0
        assert watchlist_score_nudge("OTHER", spot_file) == Decimal("0")  # symbols independent

        # a second stop-out roughly doubles it
        one = watchlist_score_nudge("EPIC", spot_file)
        record_stop_out("EPIC", spot_file)
        two = watchlist_score_nudge("EPIC", spot_file)
        assert two < one, (one, two)

        # one capital block is not enough; two crosses the threshold
        record_capital_block("SAGA", spot_file)
        assert watchlist_score_nudge("SAGA", spot_file) == Decimal("0")
        record_capital_block("SAGA", spot_file)
        assert watchlist_score_nudge("SAGA", spot_file) == WATCHLIST_CAPITAL_BONUS

        # the two feeds compose on one symbol
        record_capital_block("EPIC", spot_file)
        record_capital_block("EPIC", spot_file)
        assert watchlist_score_nudge("EPIC", spot_file) == two + WATCHLIST_CAPITAL_BONUS

        # books are isolated — a short stop-out means price ROSE, which is
        # bullish for a long, so this must not bleed across
        record_stop_out("EPIC", futures_file)
        assert list(get_watchlist(futures_file)) == ["EPIC"]
        assert set(get_watchlist(spot_file)) == {"EPIC", "SAGA"}

        # prune drops stale records and then the empty symbol entirely
        stale = (datetime.now(timezone.utc) - timedelta(hours=float(WATCHLIST_MEMORY_HOURS) + 1)).isoformat()
        _save_watchlist({"OLD": {"stop_outs": [stale], "capital_blocks": []}}, spot_file)
        prune_watchlist(spot_file)
        assert get_watchlist(spot_file) == {}, get_watchlist(spot_file)

        # pruning one book leaves the other untouched
        prune_watchlist(spot_file)
        assert list(get_watchlist(futures_file)) == ["EPIC"]
    finally:
        for f in (spot_file, futures_file):
            if os.path.exists(f):
                os.remove(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from watchlist import _test_watchlist_state; _test_watchlist_state()"`
Expected: `NameError: name 'get_watchlist' is not defined`

- [ ] **Step 3: Write the minimal implementation**

Insert into `watchlist.py` after `capital_block_bonus` and before the test functions:

```python
WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")
WATCHLIST_FILE_FUTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist_futures.json")

_EMPTY_ENTRY = {"stop_outs": [], "capital_blocks": []}


def _load_watchlist(watchlist_file: str) -> dict:
    if not os.path.exists(watchlist_file):
        return {}
    try:
        with open(watchlist_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}  # advisory state: a corrupt file must not stop trading


def _save_watchlist(entries: dict, watchlist_file: str) -> None:
    with open(watchlist_file, "w") as f:
        json.dump(entries, f)


def _record(symbol: str, field: str, watchlist_file: str) -> None:
    entries = _load_watchlist(watchlist_file)
    entry = entries.setdefault(symbol, dict(_EMPTY_ENTRY))
    entry.setdefault(field, []).append(datetime.now(timezone.utc).isoformat())
    _save_watchlist(entries, watchlist_file)


def record_stop_out(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> None:
    """Called when a position closes with exit_reason == "stop_loss"."""
    _record(symbol, "stop_outs", watchlist_file)


def record_capital_block(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> None:
    """Called when a symbol cleared every quality gate and was refused only
    for capital — i.e. any rejection AFTER `chosen` is set."""
    _record(symbol, "capital_blocks", watchlist_file)


def watchlist_score_nudge(symbol: str, watchlist_file: str = WATCHLIST_FILE) -> Decimal:
    """Combined advisory nudge for this symbol: cooldown penalty (negative)
    plus capital-block bonus (positive). Zero when the symbol is unknown."""
    entry = _load_watchlist(watchlist_file).get(symbol)
    if not entry:
        return Decimal("0")
    now = datetime.now(timezone.utc)
    return (cooldown_penalty(entry.get("stop_outs", []), now)
            + capital_block_bonus(entry.get("capital_blocks", []), now))


def prune_watchlist(watchlist_file: str = WATCHLIST_FILE) -> None:
    """Drop records older than the memory window, then drop symbols left with
    nothing. Keeps the file bounded without a separate expiry concept."""
    entries = _load_watchlist(watchlist_file)
    now = datetime.now(timezone.utc)
    kept = {}
    for symbol, entry in entries.items():
        fresh = {
            field: [t.isoformat() for t in _within_window(entry.get(field, []), now, WATCHLIST_MEMORY_HOURS)]
            for field in ("stop_outs", "capital_blocks")
        }
        if any(fresh.values()):
            kept[symbol] = fresh
    if kept != entries:
        _save_watchlist(kept, watchlist_file)


def get_watchlist(watchlist_file: str = WATCHLIST_FILE) -> dict:
    return _load_watchlist(watchlist_file)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from watchlist import _test_nudge_math, _test_watchlist_state; _test_nudge_math(); _test_watchlist_state(); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Add the report printer and test runner**

Append to the very end of `watchlist.py`:

```python
def _print_watchlist(label: str, watchlist_file: str) -> None:
    entries = get_watchlist(watchlist_file)
    print(f"=== {label} ===")
    if not entries:
        print("  (empty)")
        return
    now = datetime.now(timezone.utc)
    for symbol in sorted(entries):
        entry = entries[symbol]
        stops = len(entry.get("stop_outs", []))
        blocks = len(entry.get("capital_blocks", []))
        nudge = watchlist_score_nudge(symbol, watchlist_file)
        print(f"  {symbol:10} stop-outs={stops}  capital-blocks={blocks}  nudge={nudge:+.3f}")


if __name__ == "__main__":
    _test_nudge_math()
    _test_watchlist_state()
    _print_watchlist("SPOT WATCHLIST", WATCHLIST_FILE)
    _print_watchlist("FUTURES WATCHLIST", WATCHLIST_FILE_FUTURES)
```

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python watchlist.py`
Expected: both watchlists print as `(empty)` with no traceback.

- [ ] **Step 6: Gitignore the state files**

In `.gitignore`, find the line `reversal_exit_markers.json` and add two lines directly after it:

```
watchlist.json
watchlist_futures.json
```

- [ ] **Step 7: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add watchlist.py .gitignore
git commit -m "$(cat <<'EOF'
Add watchlist state layer and report printer

Per-book JSON files, because the books are directionally opposite: a
short stop-out means price rose, which is bullish for a long, so a
shared file would leak the inverted signal. Same reasoning already
applied to trade_peaks.json / trade_peaks_futures.json.

A missing or corrupt state file degrades to empty rather than raising —
the watchlist is advisory and must never stop a trading cycle.

`python watchlist.py` prints both watchlists. Not wired into trading
yet; that's the next commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Thread `watchlist_nudge` through `compute_opportunity_score`

**Files:**
- Modify: `execute.py` (signature at 980-983, blend at 1023, reasoning at 1046-1052, and `_test_compute_opportunity_score`)

**Interfaces:**
- Consumes: nothing from Task 1/2 directly — this task only adds a defaulted parameter
- Produces: `compute_opportunity_score(..., watchlist_nudge: Decimal = Decimal("0"))`, additive into `opportunity_score`

- [ ] **Step 1: Write the failing test**

Find `_test_compute_opportunity_score` in `execute.py` and add these lines immediately before its final `print("compute_opportunity_score self-check OK")`:

```python
    # watchlist nudge composes additively, same as the fib/fear-greed nudges
    base_wl = compute_opportunity_score(Decimal("0.60"), Decimal("1"))["opportunity_score"]
    penalised = compute_opportunity_score(Decimal("0.60"), Decimal("1"), watchlist_nudge=Decimal("-0.1"))
    assert penalised["opportunity_score"] < base_wl, "a cooldown penalty must lower the score"
    boosted = compute_opportunity_score(Decimal("0.60"), Decimal("1"), watchlist_nudge=Decimal("0.05"))
    assert boosted["opportunity_score"] > base_wl, "a capital-block bonus must raise the score"
    neutral_wl = compute_opportunity_score(Decimal("0.60"), Decimal("1"), watchlist_nudge=Decimal("0"))
    assert neutral_wl["opportunity_score"] == base_wl, "the default must be an exact no-op"
    assert "watchlist" in penalised["reasoning"].lower(), penalised["reasoning"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from execute import _test_compute_opportunity_score; _test_compute_opportunity_score()"`
Expected: `TypeError: compute_opportunity_score() got an unexpected keyword argument 'watchlist_nudge'`

- [ ] **Step 3: Write the minimal implementation**

3a. Change the signature (currently `execute.py:980-983`) from:

```python
def compute_opportunity_score(
    llm_confidence: Decimal, kronos_pct: Decimal | None,
    fear_greed: int | None = None, is_short: bool = False, fib_score: Decimal = Decimal("0"),
) -> dict:
```

to:

```python
def compute_opportunity_score(
    llm_confidence: Decimal, kronos_pct: Decimal | None,
    fear_greed: int | None = None, is_short: bool = False, fib_score: Decimal = Decimal("0"),
    watchlist_nudge: Decimal = Decimal("0"),
) -> dict:
```

3b. Add a docstring bullet. Find the `- fib_score:` bullet in that function's docstring and add directly after it:

```
    - watchlist_nudge: advisory memory of this symbol (see watchlist.py) —
      negative while a stop-out cooldown is decaying, positive when the symbol
      keeps clearing every gate only to be refused for capital. Defaults to 0
      so callers that don't consult the watchlist are unaffected.
```

3c. Change the blend line (currently `execute.py:1023`) from:

```python
    opportunity_score = KRONOS_WEIGHT * kronos_score + LLM_WEIGHT * llm_score + fib_score
```

to:

```python
    opportunity_score = KRONOS_WEIGHT * kronos_score + LLM_WEIGHT * llm_score + fib_score + watchlist_nudge
```

3d. Add the reasoning fragment. Find `fib_note = ...` (currently `execute.py:1046`) and add directly after it:

```python
    watchlist_note = f" Watchlist nudge {watchlist_nudge:+.2f}." if watchlist_nudge != 0 else ""
```

then change the reasoning line `f"{fear_greed_note}{fib_note} "` to:

```python
        f"{fear_greed_note}{fib_note}{watchlist_note} "
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "from execute import _test_compute_opportunity_score; _test_compute_opportunity_score(); print('ok')"`
Expected: `compute_opportunity_score self-check OK` then `ok`

- [ ] **Step 5: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute.py
git commit -m "$(cat <<'EOF'
Thread watchlist_nudge through compute_opportunity_score

Defaulted parameter, additive into the blend exactly like the existing
fib_score and fear_greed nudges, so every current caller is unaffected
and passing 0 is an exact no-op. Surfaced in the reasoning string
alongside the others.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire the recording hooks and the nudge into both books

**Files:**
- Modify: `execute.py` (imports; `_classify_and_log_closed_leg` around 1640; call site 1886-1888; capital blocks at 1975 and 2008)
- Modify: `execute_futures.py` (imports; `_log_closed_short` around 283; call site 500-502; capital blocks at 555 and 567)

**Interfaces:**
- Consumes: `record_stop_out`, `record_capital_block`, `watchlist_score_nudge`, `prune_watchlist`, `WATCHLIST_FILE`, `WATCHLIST_FILE_FUTURES` from Task 2; the `watchlist_nudge` parameter from Task 3

- [ ] **Step 1: Add the imports**

In `execute.py`, directly below the existing `from scanner import (...)` block, add:

```python
from watchlist import (
    WATCHLIST_FILE,
    prune_watchlist,
    record_capital_block,
    record_stop_out,
    watchlist_score_nudge,
)
```

In `execute_futures.py`, directly below its existing `from scanner import (...)` block, add:

```python
from watchlist import (
    WATCHLIST_FILE_FUTURES,
    prune_watchlist,
    record_capital_block,
    record_stop_out,
    watchlist_score_nudge,
)
```

- [ ] **Step 2: Record spot stop-outs**

In `execute.py`'s `_classify_and_log_closed_leg`, find:

```python
        exit_reason, exit_time = "stop_loss", sl_order["updateTime"]
```

and add directly after it:

```python
        record_stop_out(symbol, WATCHLIST_FILE)
```

- [ ] **Step 3: Record futures stop-outs**

In `execute_futures.py`'s `_log_closed_short`, find:

```python
        filled_at = algo_fill_price(algo)
        if filled_at is not None:
            exit_price = filled_at
            exit_reason = reason
```

and change the body to:

```python
        filled_at = algo_fill_price(algo)
        if filled_at is not None:
            exit_price = filled_at
            exit_reason = reason
            if reason == "stop_loss":
                record_stop_out(pos["symbol"], WATCHLIST_FILE_FUTURES)
```

Note: this hook only works because of the 2026-08-05 fix to `algo_fill_price`. Before it, `_log_closed_short` read `executedQty`/`avgPrice` — fields absent from Binance's algo-order response — so no futures exit was ever labeled `stop_loss` and this branch was unreachable.

- [ ] **Step 4: Pass the nudge at both call sites**

In `execute.py`, change (currently 1886-1888):

```python
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, fib_score=fib_score,
        )
```

to:

```python
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, fib_score=fib_score,
            watchlist_nudge=watchlist_score_nudge(sym, WATCHLIST_FILE),
        )
```

In `execute_futures.py`, change (currently 500-502):

```python
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, is_short=True, fib_score=fib_score,
        )
```

to:

```python
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, is_short=True, fib_score=fib_score,
            watchlist_nudge=watchlist_score_nudge(sym, WATCHLIST_FILE_FUTURES),
        )
```

- [ ] **Step 5: Record spot capital blocks**

Both spot blocks sit after `chosen` is resolved, so the qualifying symbol is `top_symbol`.

In `execute.py`, change:

```python
    if room <= 0:
        print("Not betting — already at or over the portfolio exposure cap.")
        return
```

to:

```python
    if room <= 0:
        print("Not betting — already at or over the portfolio exposure cap.")
        record_capital_block(top_symbol, WATCHLIST_FILE)
        return
```

and change:

```python
    if bet_size < min_viable_bet:
        print(f"Not betting — remaining room (${bet_size:.2f}) is below {MIN_BET_FRACTION * 100:.0f}% of the "
              f"normal ${per_trade_cap:.2f} bet size (min viable ${min_viable_bet:.2f}). Waiting for room to free up.")
        return
```

to:

```python
    if bet_size < min_viable_bet:
        print(f"Not betting — remaining room (${bet_size:.2f}) is below {MIN_BET_FRACTION * 100:.0f}% of the "
              f"normal ${per_trade_cap:.2f} bet size (min viable ${min_viable_bet:.2f}). Waiting for room to free up.")
        record_capital_block(top_symbol, WATCHLIST_FILE)
        return
```

- [ ] **Step 6: Record futures capital blocks**

Both futures blocks sit after `chosen` is resolved, so the qualifying symbol is `worst_symbol`.

In `execute_futures.py`, change:

```python
    if room <= 0:
        print("Not shorting — already at or over the futures exposure cap.")
        return
```

to:

```python
    if room <= 0:
        print("Not shorting — already at or over the futures exposure cap.")
        record_capital_block(worst_symbol, WATCHLIST_FILE_FUTURES)
        return
```

and change:

```python
    if margin < 5:
        print(f"Not shorting — available margin (${margin:.2f}) too small to be worth a trade.")
        return
```

to:

```python
    if margin < 5:
        print(f"Not shorting — available margin (${margin:.2f}) too small to be worth a trade.")
        record_capital_block(worst_symbol, WATCHLIST_FILE_FUTURES)
        return
```

Do **not** add a hook to the daily-loss-limit blocks in either file. Those are a policy halt applying to the whole book, not a judgement about the symbol.

- [ ] **Step 7: Prune once per cycle in both books**

In `execute.py`'s `manage_open_positions`, find the existing line:

```python
    prune_closed_trade_peaks({p["trade_id"] for p in positions})
```

and add directly after it:

```python
    prune_watchlist(WATCHLIST_FILE)
```

In `execute_futures.py`'s `manage_short_positions`, find:

```python
    prune_closed_trade_peaks(
        {p["trade_id"] for p in positions}, peaks_file=TRADE_PEAKS_FILE_FUTURES,
    )
```

and add directly after it:

```python
    prune_watchlist(WATCHLIST_FILE_FUTURES)
```

- [ ] **Step 8: Verify both modules import cleanly**

Run: `cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "import execute, execute_futures; print('import ok')"`
Expected: `import ok` — this also proves no circular import was introduced (`watchlist.py` imports nothing from `execute.py`, which is what keeps it acyclic).

- [ ] **Step 9: Commit**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner
git add execute.py execute_futures.py
git commit -m "$(cat <<'EOF'
Wire the watchlist into both books

Stop-outs recorded from both journal classifiers; capital blocks
recorded at the four post-`chosen` rejection paths, where the symbol
had already cleared every quality gate and was refused only for
funding. The resulting nudge feeds compute_opportunity_score at both
call sites, and each book prunes its own file once per cycle.

Daily-loss-limit halts deliberately do NOT record — that's a
book-wide policy stop, not a judgement about the symbol.

The futures stop-out hook depends on the 2026-08-05 algo_fill_price
fix; before it no futures exit was ever labeled stop_loss, so the
branch was unreachable.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Full regression, live smoke test, deploy

**Files:** none modified — verification and deployment only.

- [ ] **Step 1: Run the full regression suite**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && set -a && source .env && set +a && timeout 120 venv/bin/python -c "
import scanner
for t in ('_test_rank_symbols','_test_equal_profit_fractions','_test_pump_signature','_test_compute_rsi',
          '_test_compute_atr','_test_compute_fibonacci_levels','_test_compute_vwap','_test_compute_obv',
          '_test_detect_obv_divergence','_test_get_text_sentiment','_test_get_market_breadth',
          '_test_describe_events','_test_describe_pump_risk'):
    getattr(scanner, t)()
print('SCANNER OK')

import watchlist
watchlist._test_nudge_math(); watchlist._test_watchlist_state()
print('WATCHLIST OK')

import execute, tempfile, os
tmp = tempfile.gettempdir()
execute._test_round_step(); execute._test_should_exit_early()
execute._test_reversal_exit_markers(); execute._test_trailing_floors(); execute._test_trade_peaks()
execute._test_update_peak_and_drawdown(os.path.join(tmp,'_p.json'))
execute._test_confirm_momentum(os.path.join(tmp,'_c.json'))
execute._test_golden_trade_marking(os.path.join(tmp,'_g.json'))
execute._test_compute_opportunity_score(); execute._test_compute_atr_stop_loss_pct()
execute._test_compute_fib_stop_loss_pct(); execute._test_compute_fib_take_profit_pcts()
execute._test_fib_entry_signal(); execute._test_use_fixed_stop_loss(); execute._test_meets_min_reward_risk()
execute._test_check_daily_loss_limit(os.path.join(tmp,'_d.json'))
execute._test_compute_atr_take_profit_pcts(); execute._test_is_counter_trend()
execute._test_vwap_deviation(); execute._test_obv_warns_against()
print('EXECUTE OK')

import execute_futures
execute_futures._test_should_exit_short_early(); execute_futures._test_algo_fill_classification()
print('EXECUTE_FUTURES OK')

import backtest
backtest._test_simulate_exit(); backtest._test_sourced_stop_and_tp()
print('BACKTEST OK')
" 2>&1 | grep -v NotOpenSSLWarning | grep -v warnings.warn
```

Expected: all five `OK` lines, no traceback.

- [ ] **Step 2: Live smoke test against real state, changing nothing**

This proves the nudge path works end to end on real symbols before it can affect a trade. It writes only to a temp file.

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && venv/bin/python -c "
import tempfile, os
from decimal import Decimal
from watchlist import record_stop_out, record_capital_block, watchlist_score_nudge, get_watchlist
from execute import compute_opportunity_score

fd, f = tempfile.mkstemp(suffix='.json'); os.close(fd); os.remove(f)
try:
    # replay EPIC's real history: two stop-outs
    record_stop_out('EPIC', f); record_stop_out('EPIC', f)
    # replay a repeatedly capital-blocked symbol
    record_capital_block('SAGA', f); record_capital_block('SAGA', f)
    for sym in ('EPIC', 'SAGA', 'NEVER_SEEN'):
        n = watchlist_score_nudge(sym, f)
        r = compute_opportunity_score(Decimal('0.60'), Decimal('1'), watchlist_nudge=n)
        print(f'{sym:12} nudge={n:+.3f}  score={r[\"opportunity_score\"]:+.3f}  passes={r[\"passes\"]}')
finally:
    if os.path.exists(f): os.remove(f)
" 2>&1 | grep -v NotOpenSSLWarning | grep -v warnings.warn
```

Expected: `EPIC` shows a clearly negative nudge (~-0.2, two stop-outs at full strength) and the lowest score; `SAGA` shows `+0.050`; `NEVER_SEEN` shows `+0.000`. If EPIC's nudge is not roughly double a single stop-out's, stop and investigate rather than deploying.

- [ ] **Step 3: Restart the live loop**

`loop.py` runs the old code in memory and will not pick up any of this until restarted.

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

- [ ] **Step 4: Confirm it's live**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && tail -20 loop.log
```

Confirm a new timestamped cycle is running under the new PID with no traceback. The watchlist starts empty and accumulates, so no `Watchlist nudge` text will appear in reasoning strings until a stop-out or a repeated capital block actually occurs — its absence at this point is expected, not a failure.

- [ ] **Step 5: Push**

```bash
cd /Users/mayursachdeva/Documents/Trading/memecoin-scanner && git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** every spec section maps to a task — pure functions and constants (Task 1), state layer plus per-book files, pruning and the report printer (Task 2), the `compute_opportunity_score` parameter (Task 3), all three integration hooks (Task 4), testing and the honest-limitation check (Tasks 1-2 and 5). The spec's "out of scope" items (reordering candidate selection, auto re-entry, backfilling history, recording earlier-gate rejections) are untouched by every task.
- **Type consistency checked:** `Decimal` throughout the score path; timestamps stored and compared as timezone-aware UTC ISO strings; `watchlist_file` is the parameter name in every state function, mirroring the `peaks_file` convention already established by `prune_closed_trade_peaks`. `record_stop_out` takes `symbol` (not `trade_id`) at both hooks — spot passes `symbol` from the unpacked key, futures passes `pos["symbol"]`.
- **No placeholders:** every step contains runnable code or an exact edit with both before and after text.
