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
            dt = datetime.fromisoformat(t)
            # Timezone-naive datetimes are unreliable (no instantaneous meaning)
            # and comparing them with tz-aware dts raises TypeError in _within_window.
            # Skip rather than assume UTC, which would invent information.
            if dt.tzinfo is None:
                continue
            parsed.append(dt)
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

    # timezone-naive timestamps are unusable (no reliable instant) and must be
    # SKIPPED rather than raising into a trading cycle — a live cycle died on
    # exactly this class of input before the guard existed
    naive = "2026-08-05T11:00:00"
    assert cooldown_penalty([naive], now) == Decimal("0")
    assert capital_block_bonus([naive, ago(1)], now) == Decimal("0")
    # a naive entry alongside valid ones is dropped without disturbing them
    assert cooldown_penalty([naive, ago(1)], now) == cooldown_penalty([ago(1)], now)
    assert capital_block_bonus([naive, ago(1), ago(2)], now) == WATCHLIST_CAPITAL_BONUS
