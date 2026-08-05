# Watchlist — Advisory Entry Feed

**Date:** 2026-08-05
**Status:** Approved, not yet implemented

## Problem

Two gaps in how the bot treats symbols it has already formed an opinion about:

**1. Nothing remembers a stop-out.** A symbol that stopped us out is treated on its next evaluation exactly like one we've never traded. EPIC was stopped out twice — **-114.19 and -197.96 USDT, together 58% of all futures losses** — and the bot re-entered it a third time (2026-08-05) with nothing in the logic even noting the history.

**2. Nothing remembers a near-miss.** Symbols that clear every quality gate and are refused purely for lack of capital are discarded silently. `loop.log` is full of `Not betting — remaining room ($690.15) is below 50% of the normal $4244.80 bet size`. These are the highest-quality rejections the bot produces — its own judgment already said yes — and they leave no trace.

## Goal

A per-symbol memory that **biases** entry decisions without ever forcing one, and that is readable so the operator can see what the bot is tracking.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Role | **Advisory** — nudges the opportunity score, never triggers an entry | Contained blast radius on a live, profitable book. Same mechanism as the existing Fib and Fear&Greed nudges. |
| Stop-out direction | **Penalty** (harder to re-enter), not bonus | The literal request was "track it for another entry", but the loss data contradicts the intuitive reading: repeat entries on stopped-out symbols are the single largest loss source. The symbol stays tracked and re-enterable — it just has to clear a higher bar than the conditions that already failed. |
| Cooldown shape | **Time-decay × repeat-offender count** | Encodes "this symbol specifically has burned us repeatedly." EPIC's third attempt faces ~2× a first-timer's penalty; a flat penalty would treat them identically, which is exactly the case the data says to separate. |
| Near-miss definition | **Passed every quality gate, blocked only by capital** | Unambiguous, low-volume, and the one category where the bot's own judgment already approved the trade. Logging every rejection would be ~2,800 rows/day dominated by tokens seen once and awaiting 2-cycle confirmation — an analytics exercise, not a watchlist. |
| Near-miss nudge | **Small positive, only after ≥2 blocks in the window** | A single block is weak evidence and could be coincidence. Repeated blocking demonstrates persistent quality. |

## Design

### New module: `watchlist.py`

Its own state, its own logic, a narrow interface. `execute.py` is already ~2,000 lines; this does not belong in it. Both books consume the module.

### State

`watchlist.json` (spot) and `watchlist_futures.json` (futures), keyed by symbol:

```json
{"EPIC": {"stop_outs": ["2026-08-05T08:27:19+00:00"],
          "capital_blocks": ["2026-08-04T12:05:00+00:00"]}}
```

**Per-book files are required, not stylistic.** The books are directionally opposite: a *short* stop-out on EPIC means price rose, which is bullish information for a *long*. A shared file would leak the inverted signal across books. Same reasoning already applied to `trade_peaks.json` / `trade_peaks_futures.json` and to `check_daily_loss_limit`'s per-book file parameter.

### Pure functions (no I/O, fully testable)

```
cooldown_penalty(stop_out_times: list[str], now: datetime) -> Decimal
```
Returns `-WATCHLIST_COOLDOWN_BASE × offender_count × decay`, where `offender_count` is the number of stop-outs inside `WATCHLIST_MEMORY_HOURS` and `decay` falls linearly from 1 to 0 across `WATCHLIST_COOLDOWN_HOURS` since the most recent stop-out. Zero once decayed or when there are no stop-outs.

```
capital_block_bonus(block_times: list[str], now: datetime) -> Decimal
```
Returns `+WATCHLIST_CAPITAL_BONUS` when at least `WATCHLIST_MIN_BLOCKS` blocks fall inside `WATCHLIST_MEMORY_HOURS`, else zero.

### State functions

- `record_stop_out(symbol, watchlist_file)`
- `record_capital_block(symbol, watchlist_file)`
- `watchlist_score_nudge(symbol, watchlist_file) -> Decimal` — sum of the two pure functions above; the single value call sites pass through
- `prune_watchlist(watchlist_file)` — drops timestamps outside the memory window and symbols left with no records
- `get_watchlist(watchlist_file) -> dict` — for reporting

### Constants (env-overridable, matching existing convention)

| Constant | Default | Meaning |
|---|---|---|
| `WATCHLIST_COOLDOWN_BASE` | 0.10 | Penalty per prior stop-out at full strength. Matches `FEAR_GREED_NUDGE`/`FIB_SIGNAL_NUDGE` magnitude against a 0.3 threshold. |
| `WATCHLIST_COOLDOWN_HOURS` | 24 | Decay window back to neutral |
| `WATCHLIST_CAPITAL_BONUS` | 0.05 | Half a standard nudge — deliberately weaker than the penalty |
| `WATCHLIST_MIN_BLOCKS` | 2 | Blocks required before any bonus applies |
| `WATCHLIST_MEMORY_HOURS` | 168 | 7 days; how long records count and are retained |

### Integration — three additive hooks

1. **`compute_opportunity_score(..., watchlist_nudge: Decimal = Decimal("0"))`** — new defaulted parameter, identical additive pattern to the existing `fear_greed` and `fib_score` nudges, so every current caller is unaffected. Included in the reasoning string like the others.
2. **`record_stop_out(symbol)`** from both journal classifiers when `exit_reason == "stop_loss"`: `_classify_and_log_closed_leg` (`execute.py`) and `_log_closed_short` (`execute_futures.py`). **This depends on the 2026-08-05 fix** — before it, `_log_closed_short` read `executedQty`/`avgPrice`, fields that do not exist on Binance's algo-order response, so no futures exit was ever labeled `stop_loss` and the watchlist would have recorded nothing from that book.
3. **`record_capital_block(symbol)`** at the post-`chosen` block paths: `execute.py:1976` (exposure cap) and `:2008` (room below `MIN_BET_FRACTION`); `execute_futures.py:555` and `:567`. Clean rule: anything rejected *after* `chosen` is set qualified on merit and was refused funding.

Plus `prune_watchlist()` once per cycle per book, and a `__main__` block that prints the current watchlist so it can actually be read.

### Known limitation of the positive nudge

Candidates are iterated in **momentum-rank order** and the loop `break`s on the first qualifier. The nudge shifts pass/fail at the 0.3 threshold; it does **not** reorder selection. Its real effect is therefore narrow: it keeps a repeatedly-blocked, still-good symbol above the line when its signals soften slightly. It will not advance a symbol past higher-ranked candidates. Reordering the ranking would be a materially larger change to live-book machinery and is deliberately out of scope.

### Error handling

Every function tolerates a missing or empty state file (returns `{}` / zero nudge) rather than raising — matching `_load_trade_peaks` and `_load_reversal_exit_markers`. Malformed ISO timestamps are skipped rather than crashing a trading cycle; the watchlist is advisory, so degrading to "no nudge" is always safe.

### Testing

`_test_*` convention, no pytest, no network:
- `cooldown_penalty`: zero with no history; full strength immediately after a stop-out; ~half at the midpoint of the decay window; zero past it; scales with offender count; ignores stop-outs outside the memory window.
- `capital_block_bonus`: zero at 1 block, bonus at 2, zero when blocks fall outside the window.
- **EPIC regression:** a symbol with two stop-outs inside the window receives approximately double the penalty of one with a single stop-out.
- State lifecycle via tempfiles: record → read → prune, and per-book isolation (pruning one file must not affect the other).
- Extend `_test_compute_opportunity_score` for the new parameter: positive raises, negative lowers, default `Decimal("0")` is an exact no-op.

## Out of scope

- Reordering candidate selection by watchlist membership.
- Auto re-entry, or any path that creates a position without passing all existing gates.
- Backfilling historical stop-outs into the watchlist — it starts empty and accumulates.
- Recording rejections at gates earlier than the capital check.
